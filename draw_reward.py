import tensorflow as tf
import matplotlib.pyplot as plt

# 替换为你的.tfevents文件路径
event_file_path = 'path_to_your_events_file.tfevents....'

# 使用tf.train.summary_iterator来读取事件文件
for e in tf.train.summary_iterator(event_file_path):
    for v in e.summary.value:
        # 检查你想要的标签（例如'loss'）
        if v.tag == 'reward':
            # 提取数据（这里是scalar类型的值）
            step = e.step
            loss_value = v.simple_value
            # 使用Matplotlib绘制数据
            plt.plot(step, loss_value, label='Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Loss over Training Steps')
            plt.legend()
            plt.show()  # 你可以选择在循环外部调用plt.show()，以便一次性显示所有图表
            break  # 如果你只想绘制一个标签的数据，可以在找到后退出循环

# 注意：如果事件文件中包含多个标签的数据，并且你想在同一个图上绘制它们，
# 你需要修改上面的代码，以便在循环中累积数据，并在循环结束后绘制它们。