#Inputs:
#1. path to clean dataset
#2. list of attribute names (comma separated)
#3. Index of mutation operator

#example: python testScriptMutation.py synthetic_1_clean.csv "value" 1


#Outputs:
#1. Mutated data file in CSV
#2. A CSV file that containts outlier indexes

#assumption: The first column of dataset is an increasing index

import sys
import pandas as pd
import random 


clean_dataset=pd.read_csv(sys.argv[1])
M_dataset=clean_dataset
attribute_names=sys.argv[2].split(",")
num_of_muted_attrs=1
mutated_attributes=[0]
if len(attribute_names)>1:
    num_of_muted_attrs=random.randint(1,len(attribute_names))
    mutated_attributes=random.sample(range(1, len(attribute_names)), num_of_muted_attrs)
print("**********")
print(mutated_attributes)
print(num_of_muted_attrs)
operator=sys.argv[3]
print(operator)
#random.randint(1,5)
muted_attributes=[]

ids=[]
if operator=="1":
    #select records to mutate
    r0=int(0.002*len(M_dataset))
    mutated_records=[]
    for x in range(r0):
        mutated_records.append(random.randint(0,len(M_dataset)))
    for x in mutated_records:
        ids.append(M_dataset.ix[x, M_dataset.columns.values[0]])
else:
    #select the subsequent to mutate
    len_temp=random.randint(int(0.2*len(clean_dataset)),len(M_dataset))
    temp_df=M_dataset.tail(len_temp)
    starting_index=len(M_dataset)-len_temp
    r=random.randint(int(0.05*len(clean_dataset)) , int(0.2*len(clean_dataset)))
    for x in range(r):
        index=starting_index+x
        ids.append(M_dataset.ix[index, M_dataset.columns.values[0]])
####################################################################

for i in mutated_attributes: 
   attribute=attribute_names[i]
   muted_attributes.append(attribute)
   print(attribute)
   if operator=="1":            
    #M1
    min_value=-10*abs(int(clean_dataset[attribute].min(axis = 0)))
    max_value=10*int(clean_dataset[attribute].max(axis = 0))
    for x in mutated_records:
        index=x
        M_dataset.ix[index, attribute] = random.randint(min_value,max_value)
   if operator=="2": 
    #M2
    temp_df=M_dataset[attribute].tail(len_temp)
    shifted_horiz_df=temp_df.shift(periods=r, axis = 0)
    shifted_horiz_df=shifted_horiz_df.fillna(M_dataset.ix[index, attribute])    
    for i in range(len_temp):
        index_to_replace=i+(len(M_dataset)-len(temp_df))
        M_dataset.ix[index_to_replace,attribute]=shifted_horiz_df[index_to_replace]
   if operator=="3": 
    #M3
    temp_df=M_dataset[attribute].tail(len_temp)
    min_value=-5*abs(int(clean_dataset[attribute].min(axis = 0)))
    max_value=5*int(clean_dataset[attribute].max(axis = 0))
    value_to_add=random.randint(min_value,max_value)
    shifted_vertical_df=temp_df+value_to_add
    for i in range(r):
        index_to_replace=i+(len(M_dataset)-len(temp_df))
        M_dataset.ix[index_to_replace,attribute]=shifted_vertical_df[index_to_replace]
   if operator=="4":
    #M4
    min_value=(int(clean_dataset[attribute].min(axis = 0)))
    max_value=int(clean_dataset[attribute].max(axis = 0))
    temp_df=M_dataset[attribute].tail(len_temp)
    value_to_multiply=random.randint(min_value,max_value)
    print(temp_df)
    rescaled_df=temp_df*value_to_multiply
    print("***************")
    print(rescaled_df)
    for i in range(r):
        index_to_replace=i+(len(M_dataset)-len(temp_df))
        M_dataset.ix[index_to_replace,attribute]=rescaled_df[index_to_replace]
   if operator=="5": 
    #M5
    min_value=int(clean_dataset[attribute].min(axis = 0))
    max_value=int(clean_dataset[attribute].max(axis = 0))
    temp_df=M_dataset[attribute].tail(len_temp)
    for i in range(r):
        index_to_replace=i+(len(M_dataset)-len(temp_df))
        M_dataset.ix[index_to_replace,attribute]=random.randint(min_value,max_value)
        
M_dataset.to_csv(sys.argv[1][:-4]+"_"+''.join(map(str, muted_attributes)) +"_M"+str(operator)+".csv", index=False)
outliers_dataset = pd.DataFrame(ids, columns = [M_dataset.columns.values[0]])
outliers_dataset.to_csv(sys.argv[1][:-4]+"_"+''.join(map(str, muted_attributes))+"_M"+str(operator)+"_outliers.csv", index=False)

    
    

    

        
        
        
            
	

