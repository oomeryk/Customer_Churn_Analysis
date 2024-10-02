# gerekli kütüphanelerin import edilmesi
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# Determine the size of the dataset
num_rows = 1000

# Create a DataFrame
df = pd.DataFrame()

# 1000 satırlık random veriseti hazırlama

# unique customer number kolonunun oluşturulması
customer_numbers = set()
while len(customer_numbers) < num_rows:
    customer_number = random.randint(10000, 11000)
    customer_numbers.add(customer_number)
df['Customer_Number'] = list(customer_numbers)

# random müşteri özellikleri oluşturulması
for i in range(num_rows):
    df.loc[i, 'Age'] = random.randint(18, 65)
    df.loc[i, 'Feedback'] = random.choice(['Positive', 'Negative'])
    df.loc[i, 'year'] = random.randint(2021, 2024)
    df.loc[i, 'month'] = random.randint(1, 12)   
    df.loc[i, 'Complaint_Count'] = random.randint(1, 10)
    df.loc[i, 'Contact_Frequency'] = random.randint(1, 10)
    df.loc[i, 'Income_Level'] = random.choice(['Low', 'Medium', 'High'])
    df.loc[i, 'Recommendation'] = random.choice(['Recommend', 'Do not recommend'])
    df.loc[i, 'Purchase_Frequency'] = random.randint(1, 10)


df1 = df   
df2 = df1

#----------

# Create a LabelEncoder object
le = LabelEncoder()

# label encoding
df1["Income_Level_le"] = pd.Series(index=df1.index, dtype=int)    
for i in df1.index:
    if df1.loc[i, "Income_Level"] == "Low":
        df1.loc[i, "Income_Level_le"] = 0
    elif df1.loc[i, "Income_Level"] == "Medium":
        df1.loc[i, "Income_Level_le"] = 1
    else:
        df1.loc[i, "Income_Level_le"] = 2
        
df1['Feedback_le'] = le.fit_transform(df1['Feedback'])
df1['Recommendation_le'] = le.fit_transform(df1['Recommendation'])
df1 = df1.drop(["Customer_Number", "Recommendation", "Feedback", "Income_Level"], axis=1)

# tüm kolonları integer a çevirme
for column in df1.columns:
    df1[column] = df1[column].astype(int)

# year ve month kolonunu time kolonuna çevirme
time_list = []
for i in range(0,len(df1)):
    if df1["year"][i]==2024:
        time_list.append(12-df1["month"][i])
    else:    
        time_list.append(12-df1["month"][i]+(2024-df1["year"][i]-1)*12+12)

df1["time"] = pd.DataFrame(time_list)        
dff = df1.drop(["year", "month"], axis=1)

# tüm kolonları 0-1 aaralığına scale etme
min_max_scaler = MinMaxScaler()
data = min_max_scaler.fit_transform(dff)
data = pd.DataFrame(data)

# yeniden adlandırma
data = data.rename(columns={0:"Age",	1:"Complaint_Count",	2:"Contact_Frequency",	3:"Purchase_Frequency",	4:"Income_Level_le",	5:"Feedback_le",	6:"Recommendation_le", 7:"time"})


#---------------

data["time"] = data["time"].max()-data["time"]

# KMeans algoritma nesnesi oluşturma ve verisetimize fit etme 

km = KMeans(n_clusters=4, n_init=10)

y_pred = km.fit_predict(data)

data["segment"] = y_pred


# her segment için kolonların ortalamasını hesaplama
segment_averages = data.groupby("segment").mean()

# segment ortalamaları tablosunun her segment satırlarını toplama(ayrılma olasılığı hesaplaması için kullanılacak)
sum_list = segment_averages.sum(axis=1)

# 4 toplam sonucu arasından en büyük ve en küçük olanı bulma
min_sum = sum_list.min()
max_sum = sum_list.max()


customer_number = input("\nMüşteri numarası giriniz: ")
df["segment"] = y_pred
# segment ortalama toplamlarının min ve maxlarını kullanarak müşteri numarasına göre müşterinin ayrılma olasılığını hesaplama
customer_segment = df["segment"][df[df["Customer_Number"]==int(customer_number)].index[0]]
if sum_list[customer_segment] == min_sum:
    print("\n# Ayrılma olasılığı yüksek #\n")
elif sum_list[customer_segment] == max_sum:
    print("\n# Ayrılma olasılığı düşük #\n")
else:
    print("\n# Ayrılma olasılığı orta #\n")

    
# müşterinin özelliklerini yazdırma
customer_info = df2[df2["Customer_Number"]==int(customer_number)]
print(customer_info.drop(["Feedback_le","Recommendation_le","segment", "Income_Level_le"], axis=1).T)

