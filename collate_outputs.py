import os
import sys
import json
import glob
import pandas as pd

sys.argv= ['this', r'C:\testdata\PhantomPack']
if __name__ == '__main__':
    # collect all .json files in top_dir, recursively
    top_dir = sys.argv[1]
    glob_str = os.path.join(top_dir, '**', '*.json')
    files = glob.glob(glob_str, recursive=True)
    json_data_list = []
    for file in files:
        if "_trace" in file:
            continue # skip traceability logs
        with open(file, 'r') as f:
            json_data_list.append(json.load(f))
    
    # output in excel format
    output_file = os.path.join(top_dir, 'output.xlsx')
    df_list = []
    for data in json_data_list:
        df_row = pd.DataFrame({
            'PatientName': [data['PatientName']],
            'AcquisitionDate': [data['AcquisitionDate']],
            'AcquisitionTime': [data['AcquisitionTime']],
            'SeriesDescription_pdff': [data['SeriesDescription_pdff']],
            'SeriesNumber_pdff': [data['SeriesNumber_pdff']],
            'SeriesDescription_water': [data['SeriesDescription_water']],
            'SeriesNumber_water': [data['SeriesNumber_water']],
            'Mean 1': [data['means'][0]],
            'Mean 2': [data['means'][1]],
            'Mean 3': [data['means'][2]],
            'Mean 4': [data['means'][3]],
            'Mean 5': [data['means'][4]],
            'Stddev 1': [data['stddevs'][0]],
            'Stddev 2': [data['stddevs'][1]],
            'Stddev 3': [data['stddevs'][2]],
            'Stddev 4': [data['stddevs'][3]],
            'Stddev 5': [data['stddevs'][4]],
            'Min 1': [data['mins'][0]],
            'Min 2': [data['mins'][1]],            
            'Min 3': [data['mins'][2]],
            'Min 4': [data['mins'][3]],
            'Min 5': [data['mins'][4]],
            'Max 1': [data['maxs'][0]],
            'Max 2': [data['maxs'][1]],            
            'Max 3': [data['maxs'][2]],
            'Max 4': [data['maxs'][3]],
            'Max 5': [data['maxs'][4]],
            'Samples 1': [data['samples'][0]],
            'Samples 2': [data['samples'][1]],            
            'Samples 3': [data['samples'][2]],            
            'Samples 4': [data['samples'][3]],
            'Samples 5': [data['samples'][4]],
            'Manufacturer': [data['Manufacturer']],
            'ManufacturerModelName': [data['ManufacturerModelName']],
            'SoftwareVersions': [data['SoftwareVersions']],
            'MagneticFieldStrength': [data['MagneticFieldStrength']],
            'EchoTime': [data['EchoTime']],
            'RepetitionTime': [data['RepetitionTime']],
            'EchoTrainLength': [data['EchoTrainLength']],
            'FlipAngle': [data['FlipAngle']],
            'PulseSequenceName': [data['PulseSequenceName']],
            'InstitutionName': [data['InstitutionName']],
            'StationName': [data['StationName']],
        })
        df_list.append(df_row)
    df = pd.concat(df_list, ignore_index=True)
    df.to_excel(output_file)

