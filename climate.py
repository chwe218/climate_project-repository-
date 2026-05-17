import pandas as pd
#读取数据
train=pd.read_csv(r"data\DailyDelhiClimateTrain.csv")
# print(df.head())
test=pd.read_csv(r"data\DailyDelhiClimateTest.csv")
# print(df_test.head())

#时间
train['date']=pd.to_datetime(train['date'])
test["date"] = pd.to_datetime(test["date"])
# print(df['date'].head())
# print(df.dtypes)

#选择列：date和meantemp
train = train[["date", "meantemp"]].set_index("date")
test = test[["date", "meantemp"]].set_index("date")


#滞后特征(只记昨天)
train["yesterday_temp"] = train["meantemp"].shift(1)
test["yesterday_temp"] = test["meantemp"].shift(1)

train.dropna(inplace=True)
test.dropna(inplace=True)


#训练

from sklearn.ensemble import RandomForestRegressor
X_train = train[["yesterday_temp"]]
y_train = train["meantemp"]

X_test = test[["yesterday_temp"]]
y_test = test["meantemp"]


model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print(list(zip(y_test[:5], y_pred[:5])))


#评估
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
#平均差
mae=mean_absolute_error(y_test,y_pred)
#惩罚误差
rmse=np.sqrt(mean_squared_error(y_test,y_pred))

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))

#误差分布图
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 5))
plt.plot(test.index[:60], y_test[:60], label="Actual", linewidth=2)
plt.plot(test.index[:60], y_pred[:60], label="Predicted", linestyle="--")

plt.title("Daily Temperature: Actual vs Predicted (First 60 Days)")
plt.xlabel("Date")
plt.ylabel("Mean Temperature (°C)")

#旋转45度
plt.xticks(rotation=45)
#调整边距
plt.tight_layout()
#显示图例
plt.legend()
plt.show()

#误差图
errors = y_test - y_pred
plt.figure(figsize=(14, 3))
plt.bar(test.index, errors, color="#888")
plt.title("Forecast Errors (Actual − Predicted)")
plt.tight_layout()
plt.show()