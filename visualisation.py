 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class EDA:
        def feature_engineering(data):
            
            print(data.shape)
            
            # Headings with first 5 rows
            data.head()

            # Information on the dataset
            data.info()

            # Print data types
            print(data.dtypes)
            
            # Describe the max of each variable
            data.describe(include = 'all').T

            # FInd the total number of null in each feature
            data.isnull().sum()
                    
            #data = data.select_dtypes(include=['number'])  # Do this only if needed Keep only integer-type columns
            
            #Check again to see if all the nulls are gone 
            data.isnull().sum()
            
            # Recheck the modified data shape
            print(data.shape)
            
            choice4= int(input(" Enter to perform scaling and removal of temporal mean yes(1), no(0) "))

            if choice4==1:
                
                print("Performing temporal mean removal")
                data=EDA.remove_temporal_mean(data)
                data.isnull().sum()
                data=EDA.scale_data(data)
                data.isnull().sum()
                
            else:
                data=data
                print("No scaling")
            
            # Drop the columns which have full nan values
            data = data.dropna(axis=1, how="all")
            
            
            # Recheck again to see the sum of null values in each column
            data.isnull().sum()
            
        

            #data = data.fillna(data.median())  # Replace NaN values with column mean

            data.isnull().sum()
                    
            

            return data
            
            
        def remove_temporal_mean(df):
            """
            Removes the mean across time steps (rows) for each feature (column).
            """
            temporal_mean = df.mean(axis=0)  
            df_centered = df - temporal_mean  
            return df_centered    
        
        def scale_data(df, method="auto"):
            """
            Scales data using the specified method: "auto", "range", or "pareto".
            """
            if method == "auto":
                scaler = StandardScaler() 
            elif method == "range":
                scaler = MinMaxScaler()  
            elif method == "pareto":
                std_dev = df.std(axis=0)
                scaler = lambda x: x / np.sqrt(std_dev.replace(0, 1)) 
            else:
                raise ValueError("Invalid scaling method. Choose 'auto', 'range', or 'pareto'.")
        
            if method == "pareto":
                return df.apply(scaler, axis=0)
            else:
                return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
 
        def visualization(data):
            
            for col in data.columns:
                print(col)
                print('Skew :',round(data[col].skew(),2))
                plt.figure(figsize=(15,4))
                plt.subplot(1,2,1)
                data[col].hist(bins=10, grid=False)
                plt.ylabel('count')
                plt.subplot(1,2,2)
                sns.boxplot(x=data[col])
                plt.show()
                
            # Correlation check
            cols_list = data.select_dtypes(include = np.number).columns.tolist()

            plt.figure(figsize = (15, 7))

            sns.heatmap(
                data[cols_list].corr(numeric_only = True), annot = True, vmin = -1, vmax = 1, fmt = ".2f", cmap = "Spectral"
            )

            plt.show()
            
            corr = data.corr()

            # Finding strongly correlated variables for easier analysis
            # Create a mask to highlight values above 0.5
            mask = corr.abs() > 0.9

            # Plot the heatmap with the mask
            plt.figure(figsize=(15, 7))
            sns.heatmap(corr, annot=True, mask=~mask, cmap='coolwarm', vmin=-1, vmax=1)
            plt.show()
            
            # pairwise plotting
        
            #cols1=["Cu_GenSpeedAct__avg","In_WindSpd__avg","DynCtl_ALCmm_D__max","DynCtl_ALCmm_Q__max","MrB1_UE__max","AI_In_TowerAccelSideSideRaw__mab","AI_In_TowerAccelForeAftRaw__mab"]
            #data1=data[cols1]
            data1=data
            sns.pairplot(data1)
            plt.show()
            
            data2= EDA.correlation_feature_selection(data)
            return data2