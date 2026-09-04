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
    plt.annotate(label, (x, y), textcoords = "offset points", xytext = (0, 12.5), ha = "center", bbox = dict(boxstyle = "round,pad=0.25", alpha = 0.75, facecolor = "white", edgecolor = "gray"))
plt.xlabel("GMacs")
plt.ylabel("Dice coefficient accuracy")
plt.title("GMacs vs Dice coefficient accuracy")
plt.grid(True, alpha = 0.5)
plt.show()