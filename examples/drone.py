"""无人机示例

可以直接运行：python drone.py
演示定时感知、自动飞行和状态监控
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmind import FastMind, Graph, Event, ToolNode
from fastmind.contrib import FastMindAPI


app = FastMind()


# ============ 工具定义 ============


@app.tool(name="fly_to", description="控制无人机飞到指定坐标")
async def fly_to(x: float, y: float, z: float) -> str:
    """飞到目标位置"""
    return f"飞控: 正在飞往 ({x}, {y}, {z})"


@app.tool(name="take_photo", description="拍摄照片")
async def take_photo() -> str:
    """拍照"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"photo_{timestamp}.jpg"


@app.tool(name="get_gps", description="获取无人机当前位置")
async def get_gps() -> str:
    """获取 GPS 位置"""
    return "(37.7749, -122.4194, 100.0)"


@app.tool(name="get_battery_level", description="获取电池电量")
async def get_battery_level() -> str:
    """获取电池"""
    return "78%"


@app.tool(name="return_home", description="无人机返回起飞点")
async def return_home() -> str:
    """返回_home"""
    return "正在返回起飞点..."


# ============ 感知循环 ============


@app.perception(interval=3.0, name="flight_monitor")
async def flight_monitor(app: FastMind):
    """飞行监控感知，每3秒检查一次"""
    count = 0
    while True:
        count += 1
        yield Event(
            type="sensor.flight_data",
            payload={
                "timestamp": datetime.now().isoformat(),
                "flight_count": count,
                "battery": "78%",
                "altitude": 100.0 + (count % 10),
                "speed": 15.0,
            },
            session_id="drone_001",
        )
        await asyncio.sleep(3.0)


# ============ Agent 定义 ============


@app.agent(
    name="control_agent",
    tools=["fly_to", "take_photo", "get_gps", "get_battery_level", "return_home"],
)
async def control_agent(state: dict, event: Event) -> dict:
    """无人机控制 Agent"""
    state.setdefault("flight_log", [])
    state.setdefault("status", "idle")

    # 处理传感器数据
    if event.type == "sensor.flight_data":
        state["last_sensor_data"] = event.payload
        state["flight_log"].append(event.payload)
        if len(state["flight_log"]) > 100:
            state["flight_log"] = state["flight_log"][-100:]

        # 电池警告
        battery_str = event.payload.get("battery", "100%")
        battery = int(battery_str.replace("%", ""))
        if battery < 20 and state.get("status") != "low_battery_warning":
            state["status"] = "low_battery_warning"
            state["alert"] = "电池电量低！建议返回"

        return state

    # 处理用户命令
    if event.type == "user.message":
        text = event.payload.get("text", "")
        state["flight_log"].append({"command": text, "time": datetime.now().isoformat()})

        # 处理工具结果
        if state.get("tool_results"):
            for result in state["tool_results"]:
                state["last_result"] = result["result"]
            del state["tool_results"]
            return state

        text_lower = text.lower()

        if "位置" in text_lower or "gps" in text_lower:
            state["tool_calls"] = [
                {"id": "call_1", "function": {"name": "get_gps", "arguments": "{}"}}
            ]
        elif "电池" in text_lower or "电量" in text_lower:
            state["tool_calls"] = [
                {"id": "call_2", "function": {"name": "get_battery_level", "arguments": "{}"}}
            ]
        elif "飞" in text_lower or "fly" in text_lower:
            import re

            nums = re.findall(r"-?\d+\.?\d*", text)
            if len(nums) >= 2:
                x, y = float(nums[0]), float(nums[1])
                z = float(nums[2]) if len(nums) > 2 else 50.0
                state["tool_calls"] = [
                    {
                        "id": "call_3",
                        "function": {
                            "name": "fly_to",
                            "arguments": f'{{"x": {x}, "y": {y}, "z": {z}}}',
                        },
                    }
                ]
                state["status"] = "flying"
            else:
                state["last_result"] = "请提供目标坐标，例如: 飞往 100 200 50"
        elif "拍照" in text_lower or "photo" in text_lower:
            state["tool_calls"] = [
                {"id": "call_4", "function": {"name": "take_photo", "arguments": "{}"}}
            ]
        elif "返回" in text_lower or "home" in text_lower:
            state["tool_calls"] = [
                {"id": "call_5", "function": {"name": "return_home", "arguments": "{}"}}
            ]
            state["status"] = "returning"
        elif "状态" in text_lower or "status" in text_lower:
            sensor = state.get("last_sensor_data", {})
            status = state.get("status", "idle")
            battery = sensor.get("battery", "未知")
            altitude = sensor.get("altitude", "未知")
            state["last_result"] = f"状态: {status}, 电池: {battery}, 高度: {altitude}"
        elif "quit" in text_lower:
            state["quit"] = True
        else:
            state["last_result"] = (
                "可用命令:\n  - 位置/GPS\n  - 电池/电量\n  - 飞往 X Y Z\n  - 拍照\n  - 返回\n  - 状态"
            )

    return state


tool_node = ToolNode(app.get_tools())


def has_tool_calls(state: dict, event: Event) -> str:
    """检查是否有待执行的工具调用"""
    return "tools" if state.get("tool_calls") else None


graph = Graph()
graph.add_node("control", control_agent)
graph.add_node("tools", tool_node)

graph.add_edge("__start__", "control")

graph.add_conditional_edges("control", has_tool_calls, {"tools": "tools", None: "__end__"})
graph.add_edge("tools", "control")

graph.set_entry_point("control")
app.register_graph("main", graph)


async def main():
    fm_api = FastMindAPI(app)
    await fm_api.start()

    print("=" * 50)
    print("FastMind 无人机控制示例")
    print("=" * 50)
    print("功能:")
    print("  - 定时感知: 每3秒自动更新飞行数据")
    print("  - 位置/GPS: 查询当前位置")
    print("  - 电池/电量: 查询电池状态")
    print("  - 飞往 X Y Z: 控制飞往目标位置")
    print("  - 拍照: 拍摄照片")
    print("  - 返回: 返回起飞点")
    print("  - 状态: 查看完整状态")
    print("输入 'quit' 退出")
    print("-" * 50)
    print("提示: 无人机正在后台运行，定时发送传感器数据")
    print("-" * 50)

    session_id = "drone_001"

    # 定期转发传感器数据（事件驱动，无轮询）
    async def forward_sensor_data():
        while True:
            ev = await fm_api.wait_for_output_event("drone_001", timeout=2.0)
            if ev and ev.type == "sensor.flight_data":
                await fm_api.push_event(session_id, ev)

    forwarder = asyncio.create_task(forward_sensor_data())

    last_sensor_update = None

    while True:
        try:
            # 检查传感器数据更新
            state = fm_api.get_state(session_id)
            if state and state.get("last_sensor_data") != last_sensor_update:
                last_sensor_update = state.get("last_sensor_data")
                if last_sensor_update:
                    sensor = last_sensor_update
                    print(
                        f"\r[传感器] 时间: {sensor.get('timestamp', '')[:19]}", end="", flush=True
                    )

            # 检查警告
            if state and state.get("alert"):
                print(f"\n⚠️  {state['alert']}")
                state.pop("alert", None)

            # 尝试使用 termios，如果失败则回退到普通 input
            try:
                import select
                import sys as sys_module
                import termios
                import tty

                if sys_module.stdin.isatty():
                    old_settings = termios.tcgetattr(sys_module.stdin)
                    tty.setcbreak(sys_module.stdin.fileno())
                    try:
                        if select.select([sys_module.stdin], [], [], 0.1)[0]:
                            user_input = sys_module.stdin.readline().strip()
                        else:
                            continue
                    finally:
                        termios.tcsetattr(sys_module.stdin, termios.TCSADRAIN, old_settings)
                else:
                    # 非终端环境，使用 asyncio.to_thread 读取
                    user_input = await asyncio.to_thread(lambda: input("你: ").strip())
            except (ImportError, OSError):
                # 如果不支持，使用阻塞 input
                user_input = await asyncio.to_thread(lambda: input("你: ").strip())

            if not user_input:
                continue

            event = Event("user.message", {"text": user_input}, session_id)
            await fm_api.push_event(session_id, event)

            await asyncio.sleep(0.3)

            state = fm_api.get_state(session_id)
            if state:
                if state.get("last_result"):
                    print(f"\nBot: {state['last_result']}")

                if state.get("quit"):
                    break

        except (EOFError, KeyboardInterrupt):
            print("\n\n退出...")
            break
        except Exception as e:
            if "Bad file descriptor" not in str(e):
                print(f"\n错误: {e}")

    forwarder.cancel()
    try:
        await forwarder
    except asyncio.CancelledError:
        pass

    print("\n" + "=" * 50)
    final_state = fm_api.get_state(session_id)
    if final_state:
        print(f"飞行日志: {len(final_state.get('flight_log', []))} 条记录")
    print("=" * 50)

    await fm_api.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"需要终端支持: {e}")
        print("请使用 python -c 测试")
