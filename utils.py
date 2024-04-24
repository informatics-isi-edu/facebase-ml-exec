import os
import pandas as pd

def join_and_save_csv(base_dir, biosample_filename, genotype_filename, output_filename):
    """
    Joins two CSV files based on specific columns and saves the result to a new file.
    
    Parameters:
        base_dir (str): The base directory path.
        biosample_filename (str): The filename for the biosample CSV.
        genotype_filename (str): The filename for the genotype CSV.
        output_filename (str): The filename to save the joined table.
    """
    # Construct full paths for the files
    biosample_path = os.path.join(base_dir, biosample_filename)
    genotype_path = os.path.join(base_dir, genotype_filename)
    output_path = os.path.join(base_dir, output_filename)
    
    # Load the CSV files
    biosample_df = pd.read_csv(biosample_path)
    genotype_df = pd.read_csv(genotype_path)
    
    # Join the tables based on the 'genotype' column of biosample and 'id' column of genotype
    merged_df = pd.merge(biosample_df, genotype_df, left_on='genotype', right_on='id')
    
    # Select and rename the required columns
    final_df = merged_df[['local_identifier', 'name']]
    final_df = final_df.rename(columns={'local_identifier':'Biosample','name': 'genotype'})

    final_df['Experimental_Group'] = final_df['genotype'].apply(lambda x: 'Control' if x.endswith('+/+') else 'Experiment')

    # Save the final dataframe to a new CSV file
    final_df.to_csv(output_path, index=False)
    return final_df, output_path
