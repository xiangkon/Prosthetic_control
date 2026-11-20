"""Chinese translations of TUM trial_info task labels (docs/TUM_trial_tasks.md)."""

TUM_TASKS_ZH = (
    '初始放松姿态', 'OK 手势', '拇指向下', '欢呼／庆祝手势', '搭便车手势',
    '用手遮挡面部阳光', '打招呼', '军礼', '停止手势', '指向手势', '安静／嘘手势',
    '伸手抓取小手提箱', '伸手抓取玻璃杯', '伸手抓取电话听筒', '伸手抓取书',
    '伸手抓取小杯子', '伸手抓取苹果', '伸手抓取帽子', '伸手抓取杯子',
    '接过他人递来的托盘', '伸手抓取锁中的钥匙', '伸手抓取瓶子', '伸手抓取智能手机',
    '伸手抓取牙刷', '伸手抓取笔记本电脑', '伸手抓取笔（pen）', '伸手抓取铅笔',
    '伸手抓取茶包', '伸手抓取门把手', '伸手抓取网球', '伸手抓取瓶盖',
    '眼睛垂直运动', '眼睛水平运动', '睁眼／闭眼',
)


def tum_task_description_zh(task):
    return TUM_TASKS_ZH[task] if 0 <= task < len(TUM_TASKS_ZH) else f'未知任务（Task {task}）'
