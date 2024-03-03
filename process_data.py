import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SAMPLING_FREQUENCY = 200 # Hz
SAMPLING_PERIOD = 1 / SAMPLING_FREQUENCY

# Experiments
EYES_OPEN_RESTING = 1
EYES_CLOSED_RESTING = 2
NATIVE_IN_EAR = 5
NON_NATIVE_IN_EAR = 6
NEUTRAL_IN_EAR = 7
NATIVE_BONE = 8
NON_NATIVE_BONE = 9
NEUTRAL_BONE = 10

class Subject:
    def __init__(self, id, age, gender, smoke, alcohol, medication, language, BMI, comments):
        self.id = id
        self.age = age
        self.gender = gender
        self.smoke = smoke
        self.alcohol = alcohol
        self.medication = medication
        self.language = language
        self.BMI = BMI
        self.comments = comments if comments != "nan" else ""

    def __repr__(self):
        return f"""Subject {self.id[1:]}:
\tAge: {self.age}
\tGender: {self.gender}
\tSmoke: {self.smoke}
\tAlcohol: {self.alcohol}
\tMedication: {self.medication}
\tLanguage: {self.language}
\tBMI: {self.BMI}
\tComments: {self.comments}
"""


# Returns the start and end time for segmented data
def get_start_time(subject_id: int, experiment, session):
    df = pd.read_csv("dataset/Data_trim.csv")
    i = 0
    while df["Subject"][i] != subject_id:
        i += 1
    while df["Experment"][i] != experiment:
        i += 1
    while df["session"][i] != max(session, 1):
        i += 1
    return df["From (s)"][i]


class EEG_data:
    def __init__(self, subject, experiment, session=0, type="Filtered_Data"):
        filepath = f"dataset/{type}/{subject.id.lower()}_ex{str(experiment).zfill(2)}"
        if session != 0:
            filepath += f"_s{str(session).zfill(2)}"
        filepath += ".txt" if type == "Raw_Data" else ".csv"

        df = pd.read_csv(filepath)

        # TODO: implement support for raw_data if needed
        if type == "Raw_Data":
            print("Support for raw data not yet included")
            quit()

        else:
            self.P4 = np.array(df["P4"])
            self.Cz = np.array(df["Cz"])
            self.F8 = np.array(df["F8"])
            self.T7 = np.array(df["T7"])

        if type == "Segmented_Data":
            start_time = get_start_time(int(str(subject.id)[1:]), experiment, session)
            self.time = np.array([start_time + i*SAMPLING_PERIOD for i in range(len(self.P4))])

        else:
            self.time = np.array([i*SAMPLING_PERIOD for i in range(len(self.P4))])


def read_subjects():
    df = pd.read_csv("dataset/Subjects.csv")
    subjects = []
    for i in range(len(df["Subject ID"])):
        subjects.append(Subject(
            df["Subject ID"][i],
            int(df["Age "][i]),
            df["Gender"][i],
            df["Smoke"][i]=="Yes",
            df["Alcoholic"][i],
            df["Medication"][i]=="Yes",
            df["Mother Language"][i],
            float(df["BMI"][i]),
            df["comments"].astype(str)[i]
        ))
    return subjects


def main():
    subjects = read_subjects()

    for subject in subjects:
        data_native = EEG_data(subject, NATIVE_BONE, type="Segmented_Data")
        data_non_native = EEG_data(subject, NON_NATIVE_BONE, type="Segmented_Data")

        plt.plot(data_native.time, data_native.T7, label="Native Language")
        plt.plot(data_non_native.time, data_non_native.T7, label="Non-Native Language")

        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()