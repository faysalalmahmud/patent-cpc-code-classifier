import pandas as pd
import pathlib
from pathlib import Path

def main():
    # Load the CSV file
    csv_file_path = "data/scraped/"
    files = [
        'patent_details_20251001-102339.csv',
        'patent_details_20251001-102354.csv',
        'patent_details_20251001-102414.csv',
        'patent_details_20251001-102423.csv',
        'patent_details_20251001-102444.csv',
        'patent_details_20251001-102500.csv',
        'patent_details_20251001-102516.csv',
        'patent_details_20251001-102533.csv',
        'patent_details_20251001-102549.csv',
        'patent_details_20251001-102606.csv',
        'patent_details_20251001-102619.csv',
        'patent_details_20251001-102636.csv',
        'patent_details_20251001-102651.csv',
        'patent_details_20251001-102706.csv',
        'patent_details_20251001-102718.csv',
        'patent_details_20251001-102736.csv',
        'patent_details_20251001-102758.csv',
        'patent_details_20251001-102808.csv',
        'patent_details_20251001-102824.csv',
        'patent_details_20251001-102842.csv',
        ]
    

    dataframes_list = [pd.read_csv(csv_file_path + file) for file in files]
    df = pd.concat(dataframes_list, ignore_index=True)
    print(f"Initial shape: {df.shape}")
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df.head(3))
    df['cpc_codes'] = [eval(cpc_code) for cpc_code in df['cpc_codes']]
    print(type(df['cpc_codes'][0]))
    print(f'Shape after dropping duplicates: {df.shape}')


    # Save the cleaned DataFrame to a new CSV file
    output_path = Path("data/processed")
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_csv('data/processed/processed_patent_details.csv', index=False)



if __name__ == "__main__":
    main()