"""
消息提醒模块
Windows 桌面弹窗通知
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


def send_notification(title: str, message: str, icon_path: str = None):
    """
    发送 Windows 桌面通知

    参数:
        title: 通知标题
        message: 通知内容
        icon_path: 图标路径（可选）
    """
    if not config.ENABLE_NOTIFICATION:
        return

    try:
        from winotify import Notification, audio

        toast = Notification(
            app_id="A股量化助手",
            title=title,
            msg=message,
            duration="long",
        )

        # 设置提示音
        toast.set_audio(audio.Default, loop=False)

        toast.show()

    except ImportError:
        print(f"[通知] {title}: {message}")
    except Exception as e:
        print(f"[通知发送失败] {e}")
        print(f"[通知] {title}: {message}")


def notify_buy_signal(stock_code: str, stock_name: str, price: float, rsi: float = 0):
    """发送买入信号通知"""
    title = f"🟢 买入信号 - {stock_name}({stock_code})"
    message = f"金叉买入！当前价: ¥{price:.2f}，RSI: {rsi:.1f}"
    send_notification(title, message)


def notify_sell_signal(stock_code: str, stock_name: str, price: float, rsi: float = 0):
    """发送卖出信号通知"""
    title = f"🔴 卖出信号 - {stock_name}({stock_code})"
    message = f"死叉卖出！当前价: ¥{price:.2f}，RSI: {rsi:.1f}"
    send_notification(title, message)


def notify_trade_result(message: str):
    """发送交易结果通知"""
    send_notification("📊 交易通知", message)


if __name__ == '__main__':
    print("测试 Windows 桌面通知...")
    send_notification(
        "A股量化助手",
        "系统启动成功！正在监控您的股票池..."
    )
    notify_buy_signal('600519', '贵州茅台', 1750.00, 45.2)
    print("通知已发送，请查看 Windows 右下角")

