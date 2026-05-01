import os
import re
import json
import datetime
import asyncio
import traceback
from astrbot.api.all import *
from astrbot.api.provider import LLMResponse


@register("chat_logger", "云巫", "对话记录与检索插件", "1.0.0")
class ChatLoggerPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.base_dir = os.path.join("data", "chat_logs")
        os.makedirs(self.base_dir, exist_ok=True)

        # 异步批处理队列：攒一批消息再调用LLM，避免每条消息都打一次
        self._log_queue: asyncio.Queue = asyncio.Queue()
        self._batch_size = 10       # 每10条处理一次
        self._batch_timeout = 30    # 或30秒自动处理
        self._batch_task: asyncio.Task | None = None

    async def initialize(self):
        """插件启动时开启批处理后台任务"""
        self._batch_task = asyncio.create_task(self._batch_worker())

    async def terminate(self):
        """插件关闭时处理剩余队列并取消后台任务"""
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._flush_queue()
            except Exception:
                pass

    # ─── 存储路径 ───

    def _get_log_file_path(self, group_id: str, date_str: str) -> str:
        group_dir = os.path.join(self.base_dir, str(group_id))
        os.makedirs(group_dir, exist_ok=True)
        return os.path.join(group_dir, f"{date_str}.jsonl")

    # ─── LLM 标签识别 ───

    async def _get_chat_provider_id(self) -> str | None:
        """获取当前配置的聊天模型 Provider ID"""
        try:
            provider = self.context.get_using_provider()
            if provider:
                return provider.meta().id
        except Exception:
            pass
        # 降级：遍历所有可用provider
        providers = self.context.get_all_providers()
        if providers:
            return providers[0].meta().id
        return None

    async def _analyze_batch_with_llm(self, entries: list[dict]) -> list[dict]:
        """批量调用LLM识别标签和权重，失败则降级"""
        provider_id = await self._get_chat_provider_id()
        if not provider_id:
            logger.warning("chat_logger: 未找到可用LLM Provider，全部降级为默认标签")
            for entry in entries:
                entry["tags"] = ["未分类"]
                entry["weight"] = 3
            return entries

        prompt = """请分析以下对话内容，为每条记录提取标签和权重。
- 标签 (tags): 字符串数组（如 ["日常", "开发", "闲聊"]），支持多标签
- 权重 (weight): 1-5的整数。1-2为低优先级(闲聊等)，3为默认普通对话，4-5为高优先级(开发决策/重要约定等)

请严格以JSON数组格式返回，与输入顺序一一对应，不要有多余文字：
[{"tags": ["标签1"], "weight": 3}, ...]

对话记录：
"""
        for i, entry in enumerate(entries):
            prompt += f"\n[{i}] {entry['sender_name']}: {entry['message']}"

        try:
            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
            )
            raw = resp.completion_text.strip()
            # 正则提取最外层[]，防止模型返回带废话的markdown
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                raw = match.group(0)
            else:
                raw = raw.removeprefix("```json").removesuffix("```").strip()
            parsed = json.loads(raw)

            for i, entry in enumerate(entries):
                if i < len(parsed):
                    tags = parsed[i].get("tags", ["未分类"])
                    weight = int(parsed[i].get("weight", 3))
                    weight = max(1, min(5, weight))
                    entry["tags"] = tags
                    entry["weight"] = weight
                else:
                    entry["tags"] = ["未分类"]
                    entry["weight"] = 3
            return entries
        except Exception as e:
            logger.error(f"chat_logger: LLM批量提取失败，降级处理 - {e}")
            for entry in entries:
                entry["tags"] = ["未分类"]
                entry["weight"] = 3
            return entries

    # ─── 批处理 Worker ───

    async def _batch_worker(self):
        """后台协程：攒够数量或超时后批量处理"""
        loop = asyncio.get_event_loop()
        while True:
            try:
                batch = []
                # 1. 阻塞等待第一条消息，拿到后才开始本轮计时
                first_item = await self._log_queue.get()
                batch.append(first_item)

                # 2. 设定本轮批处理的截止时间
                deadline = loop.time() + self._batch_timeout

                # 3. 继续等后续消息，凑满 batch_size 或到时间为止
                while len(batch) < self._batch_size:
                    timeout = deadline - loop.time()
                    if timeout <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._log_queue.get(), timeout=timeout
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                # LLM打标签
                batch = await self._analyze_batch_with_llm(batch)

                # 按日期写文件
                grouped: dict[str, list] = {}
                for entry in batch:
                    key = (entry.pop("_group_id"), entry.pop("_date_str"))
                    grouped.setdefault(key, []).append(entry)

                for (gid, date_str), entries in grouped.items():
                    file_path = self._get_log_file_path(gid, date_str)
                    with open(file_path, "a", encoding="utf-8") as f:
                        for e in entries:
                            f.write(json.dumps(e, ensure_ascii=False) + "\n")

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"chat_logger: batch worker error - {e}")
                await asyncio.sleep(5)

    async def _flush_queue(self):
        """关闭时把队列剩余的刷出去"""
        remaining = []
        while not self._log_queue.empty():
            try:
                remaining.append(self._log_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if remaining:
            remaining = await self._analyze_batch_with_llm(remaining)
            grouped: dict[str, list] = {}
            for entry in remaining:
                key = (entry.pop("_group_id"), entry.pop("_date_str"))
                grouped.setdefault(key, []).append(entry)
            for (gid, date_str), entries in grouped.items():
                file_path = self._get_log_file_path(gid, date_str)
                with open(file_path, "a", encoding="utf-8") as f:
                    for e in entries:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ─── 消息监听 ───

    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def log_group_message(self, event: AstrMessageEvent):
        await self._enqueue_message(event)

    @event_message_type(EventMessageType.FRIEND_MESSAGE)
    async def log_friend_message(self, event: AstrMessageEvent):
        await self._enqueue_message(event)

    # ─── Bot 回复监听 ───

    @filter.on_llm_response()
    async def log_bot_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """监听bot的LLM回复，也记录到日志"""
        if not resp.completion_text:
            return

        msg_obj = event.message_obj
        umo = event.unified_msg_origin
        if msg_obj.group_id:
            session_key = str(msg_obj.group_id)
        else:
            sender_id = str(msg_obj.sender.user_id) if msg_obj.sender else "unknown"
            session_key = f"private_{sender_id}"

        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        entry = {
            "_group_id": session_key,
            "_date_str": date_str,
            "time": time_str,
            "sender_id": "bot",
            "sender_name": "Kei",
            "message": resp.completion_text,
            "tags": [],       # 占位，等LLM填充
            "weight": 3,
        }

        await self._log_queue.put(entry)

    async def _enqueue_message(self, event: AstrMessageEvent):
        """将消息入队，等待批量处理"""
        msg_obj = event.message_obj
        message_text = event.message_str
        sender_id = str(msg_obj.sender.user_id) if msg_obj.sender else "unknown"
        sender_name = msg_obj.sender.nickname if msg_obj.sender else "unknown"

        # 判断会话来源
        umo = event.unified_msg_origin
        if msg_obj.group_id:
            session_key = str(msg_obj.group_id)
        else:
            session_key = f"private_{sender_id}"

        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        entry = {
            "_group_id": session_key,
            "_date_str": date_str,
            "time": time_str,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "message": message_text,
            "tags": [],       # 占位，等LLM填充
            "weight": 3,
        }

        await self._log_queue.put(entry)

    # ─── 查询指令 ───

    @command("查询日志")
    async def search_logs(
        self,
        event: AstrMessageEvent,
        target_date: str = "",
        target_tag: str = "",
        min_weight: int = 0,
    ):
        """查询对话日志。用法：查询日志 [日期] [标签] [最低权重]
        示例：查询日志 2026-04-24 开发 4
        """
        umo = event.unified_msg_origin
        msg_obj = event.message_obj

        if msg_obj.group_id:
            session_key = str(msg_obj.group_id)
        else:
            sender_id = str(msg_obj.sender.user_id) if msg_obj.sender else "unknown"
            session_key = f"private_{sender_id}"

        if not target_date:
            target_date = datetime.datetime.now().strftime("%Y-%m-%d")

        file_path = self._get_log_file_path(session_key, target_date)
        if not os.path.exists(file_path):
            yield event.plain_result(f"未找到 {target_date} 的日志。")
            return

        results = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if target_tag and target_tag not in entry.get("tags", []):
                    continue
                if entry.get("weight", 3) < min_weight:
                    continue
                results.append(entry)

        if not results:
            yield event.plain_result(f"{target_date} 没有符合条件的结果。")
            return

        out_msg = f"📅 {target_date} 检索结果 (共{len(results)}条):\n"
        for r in results[:15]:
            tags_str = "/".join(r.get("tags", ["未分类"]))
            msg_preview = r.get("message", "")[:30]
            out_msg += f"[{r['time']}] {r['sender_name']}: {msg_preview} [🏷️{tags_str} | ⭐{r['weight']}]\n"
        if len(results) > 15:
            out_msg += f"\n...还有{len(results)-15}条，请用「导出日志」查看完整版"

        yield event.plain_result(out_msg)

    # ─── 导出指令 ───

    @command("导出日志")
    async def export_logs(self, event: AstrMessageEvent, target_date: str = ""):
        """导出对话日志为Markdown。用法：导出日志 [日期]"""
        if not target_date:
            target_date = datetime.datetime.now().strftime("%Y-%m-%d")

        msg_obj = event.message_obj
        if msg_obj.group_id:
            session_key = str(msg_obj.group_id)
        else:
            sender_id = str(msg_obj.sender.user_id) if msg_obj.sender else "unknown"
            session_key = f"private_{sender_id}"

        file_path = self._get_log_file_path(session_key, target_date)
        if not os.path.exists(file_path):
            yield event.plain_result(f"未找到 {target_date} 的日志。")
            return

        md_content = f"# {target_date} 会话 {session_key} 对话记录\n\n"
        md_content += "| 时间 | 发送者 | 消息内容 | 标签 | 权重 |\n"
        md_content += "| :--- | :--- | :--- | :--- | :--- |\n"

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tags_str = ", ".join(r.get("tags", []))
                safe_msg = str(r.get("message", "")).replace("\n", " ").replace("|", "\\|")
                md_content += f"| {r['time']} | {r['sender_name']} | {safe_msg} | `{tags_str}` | {r['weight']} |\n"

        export_path = os.path.join(self.base_dir, session_key, f"{target_date}_export.md")
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # 尝试以文件方式发送
        try:
            yield event.chain_result([File(export_path)])
        except Exception:
            yield event.plain_result(f"日志已导出：{export_path}")

    # ─── 标签列表指令 ───

    @command("日志标签")
    async def list_tags(self, event: AstrMessageEvent, target_date: str = ""):
        """查看某天的所有标签。用法：日志标签 [日期]"""
        if not target_date:
            target_date = datetime.datetime.now().strftime("%Y-%m-%d")

        msg_obj = event.message_obj
        if msg_obj.group_id:
            session_key = str(msg_obj.group_id)
        else:
            sender_id = str(msg_obj.sender.user_id) if msg_obj.sender else "unknown"
            session_key = f"private_{sender_id}"

        file_path = self._get_log_file_path(session_key, target_date)
        if not os.path.exists(file_path):
            yield event.plain_result(f"未找到 {target_date} 的日志。")
            return

        all_tags: set[str] = set()
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    for tag in entry.get("tags", []):
                        all_tags.add(tag)
                except json.JSONDecodeError:
                    continue

        if not all_tags:
            yield event.plain_result(f"{target_date} 没有标签记录。")
            return

        tags_sorted = sorted(all_tags)
        yield event.plain_result(f"🏷️ {target_date} 的标签：{', '.join(tags_sorted)}")