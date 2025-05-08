import pandas as pd
from typing import List, Dict, Optional
import os

class DoctorRecommender:
    def __init__(self, excel_path: str):
        """Initialize the recommender with the path to the Excel file."""
        self.excel_path = excel_path
        self.hospital_data = None
        self.load_data()
    
    def load_data(self):
        """Load hospital and doctor data from Excel file."""
        try:
            print("\n=== Loading Excel Data ===")
            # Read the Excel file
            self.hospital_data = pd.read_excel(self.excel_path)
            print(f"\nTotal rows in Excel: {len(self.hospital_data)}")
            
            # Clean up column names by removing extra spaces and unnamed columns
            self.hospital_data.columns = self.hospital_data.columns.str.strip()
            # Keep only relevant columns
            relevant_columns = ['S.NO.', 'NAME OF THE HOSPITAL', 'CONTACT INFO. (+91)', 
                              'DOCTOR RECOMMENDED', 'ASSOCIATED CATEGORY']
            self.hospital_data = self.hospital_data[relevant_columns]
            
            # Clean up the data
            # Remove rows where all columns are NaN
            self.hospital_data = self.hospital_data.dropna(how='all')
            
            # Clean up hospital names
            self.hospital_data['NAME OF THE HOSPITAL'] = self.hospital_data['NAME OF THE HOSPITAL'].str.strip()
            
            # Clean up specializations
            self.hospital_data['ASSOCIATED CATEGORY'] = self.hospital_data['ASSOCIATED CATEGORY'].str.upper()
            # Fix common typos in specializations
            specialization_fixes = {
                'OPTALMOLOGIST': 'OPHTHALMOLOGIST',
                'ORTHOPAEDIC': 'ORTHOPEDICS',
                'GYNAECHOLOGIST': 'GYNECOLOGIST',
                'ALL TYPES AVAILABLE': 'GENERAL MEDICINE'
            }
            self.hospital_data['ASSOCIATED CATEGORY'] = self.hospital_data['ASSOCIATED CATEGORY'].replace(specialization_fixes)
            
            print("\nColumns in Excel:")
            print(self.hospital_data.columns.tolist())
            
            print("\nSample of cleaned data (first 3 rows):")
            print(self.hospital_data.head(3).to_string())
            
            print("\nUnique specializations in data:")
            print(sorted(self.hospital_data['ASSOCIATED CATEGORY'].dropna().unique()))
            
            print("\nSuccessfully loaded and cleaned data from Excel file")
            
        except Exception as e:
            print(f"Error loading Excel file: {str(e)}")
            raise
    
    def get_recommendations(self, diagnosis: str, location: Optional[str] = None, 
                          specialization: Optional[str] = None) -> Dict:
        """
        Get hospital and doctor recommendations based on diagnosis and optional filters.
        
        Args:
            diagnosis: The medical diagnosis
            location: Optional location filter
            specialization: Optional doctor specialization filter
            
        Returns:
            Dictionary containing recommended hospitals and doctors
        """
        try:
            print(f"\n=== Processing Recommendations for: {diagnosis} ===")
            
            # Filter hospitals based on specialization/category
            hospitals = self.hospital_data.copy()
            print(f"\nInitial number of hospitals: {len(hospitals)}")
            
            # Map common conditions to relevant specializations (only using available categories)
            condition_to_specialization = {
                'tumor': ['SURGEON', 'DERMATOLOGIST', 'ORTHOPAEDIC'],
                'cancer': ['SURGEON', 'DERMATOLOGIST', 'ORTHOPAEDIC'],
                'fever': ['DERMATOLOGIST', 'NEUROLOGIST', 'CARDIOLOGIST'],
                'flu': ['DERMATOLOGIST', 'NEUROLOGIST', 'CARDIOLOGIST'],
                'cold': ['DERMATOLOGIST', 'NEUROLOGIST', 'CARDIOLOGIST'],
                'headache': ['NEUROLOGIST', 'PSYCHOLOGIST'],
                'heart': ['CARDIOLOGIST'],
                'kidney': ['NEPHROLOGIST'],
                'bone': ['ORTHOPAEDIC', 'SURGEON'],
                'joint': ['ORTHOPAEDIC', 'SURGEON'],
                'skin': ['DERMATOLOGIST'],
                'eye': ['OPHTHALMOLOGIST'],
                'ear': ['NEUROLOGIST', 'PSYCHOLOGIST'],
                'nose': ['NEUROLOGIST', 'PSYCHOLOGIST'],
                'throat': ['NEUROLOGIST', 'PSYCHOLOGIST'],
                'dental': ['DENTIST'],
                'mental': ['PSYCHIATRIST', 'PSYCHOLOGIST'],
                'pregnancy': ['GYNAECOLOGIST'],
                'child': ['DERMATOLOGIST', 'NEUROLOGIST'],
                'thyroid': ['NEUROLOGIST', 'PSYCHOLOGIST', 'DERMATOLOGIST'],
                'hormone': ['NEUROLOGIST', 'PSYCHOLOGIST', 'DERMATOLOGIST'],
                'diabetes': ['NEUROLOGIST', 'PSYCHOLOGIST', 'DERMATOLOGIST'],
                'metabolism': ['NEUROLOGIST', 'PSYCHOLOGIST', 'DERMATOLOGIST'],
                'gland': ['NEUROLOGIST', 'PSYCHOLOGIST', 'DERMATOLOGIST'],
                'endocrine': ['NEUROLOGIST', 'PSYCHOLOGIST', 'DERMATOLOGIST']
            }
            
            # Determine relevant specializations based on diagnosis
            relevant_specializations = []
            diagnosis_lower = diagnosis.lower()
            for condition, specializations in condition_to_specialization.items():
                if condition in diagnosis_lower:
                    relevant_specializations.extend(specializations)
            print(f"\nRelevant specializations found: {relevant_specializations}")
            
            # If we found relevant specializations, filter by them
            filtered_hospitals = pd.DataFrame()
            if relevant_specializations:
                for spec in relevant_specializations:
                    filtered_hospitals = pd.concat([
                        filtered_hospitals,
                        hospitals[hospitals['ASSOCIATED CATEGORY'].str.contains(spec, case=False, na=False)]
                    ])
                print(f"\nHospitals after specialization filter: {len(filtered_hospitals)}")
            
            # If user provided specialization, use that instead
            if specialization:
                filtered_hospitals = hospitals[hospitals['ASSOCIATED CATEGORY'].str.contains(specialization, case=False, na=False)]
                print(f"\nHospitals after user specialization filter: {len(filtered_hospitals)}")
            
            # If still no matches, include hospitals with empty ASSOCIATED CATEGORY (general/multi-specialty)
            if filtered_hospitals.empty:
                print("\nNo matches found, including hospitals with empty ASSOCIATED CATEGORY (general/multi-specialty)")
                filtered_hospitals = hospitals[hospitals['ASSOCIATED CATEGORY'].isna() | (hospitals['ASSOCIATED CATEGORY'].str.strip() == '')]
            
            # Remove rows with missing critical information
            filtered_hospitals = filtered_hospitals.dropna(subset=['NAME OF THE HOSPITAL', 'DOCTOR RECOMMENDED', 'CONTACT INFO. (+91)'])
            print(f"\nHospitals after removing missing data: {len(filtered_hospitals)}")
            
            # Remove duplicate doctors
            filtered_hospitals = filtered_hospitals.drop_duplicates(subset=['DOCTOR RECOMMENDED'])
            print(f"\nHospitals after removing duplicates: {len(filtered_hospitals)}")
            
            # Get top 3 recommendations
            top_hospitals = filtered_hospitals.head(3).to_dict('records')
            
            # Format the recommendations
            formatted_recommendations = []
            for hospital in top_hospitals:
                category = hospital['ASSOCIATED CATEGORY']
                if pd.isna(category) or category.strip() == '':
                    category = 'All Specialties / Multi-specialty'
                recommendation = {
                    'hospital_name': hospital['NAME OF THE HOSPITAL'],
                    'contact': hospital['CONTACT INFO. (+91)'],
                    'doctor': hospital['DOCTOR RECOMMENDED'],
                    'category': category
                }
                formatted_recommendations.append(recommendation)
            
            print("\nFinal recommendations:")
            for rec in formatted_recommendations:
                print(f"\nHospital: {rec['hospital_name']}\nDoctor: {rec['doctor']}\nSpecialization: {rec['category']}\nContact: {rec['contact']}")
            
            return {
                'recommendations': formatted_recommendations
            }
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return {'recommendations': []} 