import os
import re
import matplotlib.pyplot as plt

macs_list = []
accuracy_list = []
labels = []

macs_pattern = re.compile(r"([\d.]+)\s*GMac")
accuracy_pattern = re.compile(r"accuracy of ([\d.]+)")
for log_name in os.listdir("logs"):
    with open(f"logs/{log_name}", "r") as log:
        content = log.read()
        macs_match = macs_pattern.search(content)
        acc_match = accuracy_pattern.search(content)
        if macs_match and acc_match:
            macs = float(macs_match.group(1))
            acc = float(acc_match.group(1)[:-1])
            macs_list.append(macs)
            accuracy_list.append(acc)
            labels.append(log_name[:-4])
plt.figure(figsize = (8, 6))
plt.scatter(macs_list, accuracy_list, c = "black", marker = "o")
for x, y, label in zip(macs_list, accuracy_list, labels):
    plt.annotate(label, (x, y), textcoords = "offset points", xytext = (5, 5), ha = "left")
plt.xlabel("GMacs")
plt.ylabel("Dice Coefficient Accuracy")
plt.title("Model Complexity vs Accuracy")
plt.grid(True)
plt.show()