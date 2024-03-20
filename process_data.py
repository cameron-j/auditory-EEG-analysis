import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import fft

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

colours = ["C" + str(i) for i in range(0, 10)]
colour_count = 0

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

    def get_signal(self, position):
        if position == "P4":
            return self.P4
        elif position == "Cz":
            return self.Cz
        elif position == "F8":
            return self.F8
        elif position == "T7":
            return self.T7
        else:
            print(f"[ ERROR ] : {position} is not a valid EEG placement, must be [P4, Cz, F8, T7]")
            quit()

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
    run = True

    while run:
        subject_id = "S" + input("Subject:\n\t> ").zfill(2)
        placement = input("EEG placement (P4, Cz, F8 or T7):\n\t> ")

        subject = list(filter(lambda s: s.id==subject_id, subjects))[0]
        print(subject)

        native = EEG_data(subject, NATIVE_BONE, type="Segmented_Data")
        non_native = EEG_data(subject, NON_NATIVE_BONE, type="Segmented_Data")
        neutral = EEG_data(subject, NEUTRAL_BONE, type="Segmented_Data")
        eyes_closed_rest = EEG_data(subject, EYES_CLOSED_RESTING, type="Segmented_Data", session=2)
        eyes_open_rest = EEG_data(subject, EYES_OPEN_RESTING, type="Segmented_Data", session=2)

        native_ft = np.abs(np.fft.fft(native.get_signal(placement) - np.average(native.get_signal(placement))))
        native_ft_freq = np.fft.fftfreq(len(native.get_signal(placement)), d=SAMPLING_PERIOD)
        non_native_ft = np.abs(np.fft.fft(non_native.get_signal(placement) - np.average(non_native.get_signal(placement))))
        non_native_ft_freq = np.fft.fftfreq(len(non_native.get_signal(placement)), d=SAMPLING_PERIOD)
        neutral_ft = np.abs(np.fft.fft(neutral.get_signal(placement) - np.average(neutral.get_signal(placement))))
        neutral_ft_freq = np.fft.fftfreq(len(neutral.get_signal(placement)), d=SAMPLING_PERIOD)
        eyes_closed_rest_ft = np.abs(np.fft.fft(eyes_closed_rest.get_signal(placement) - np.average(eyes_closed_rest.get_signal(placement))))
        eyes_closed_rest_ft_freq = np.fft.fftfreq(len(eyes_closed_rest.get_signal(placement)), d=SAMPLING_PERIOD)
        eyes_open_rest_ft = np.abs(np.fft.fft(eyes_open_rest.get_signal(placement) - np.average(eyes_open_rest.get_signal(placement))))
        eyes_open_rest_ft_freq = np.fft.fftfreq(len(eyes_open_rest.get_signal(placement)), d=SAMPLING_PERIOD)

        fig, ax = plt.subplots(1, 2)
        fig.suptitle(f"{placement} EEG data for auditory stimulus\nSubject {subject.id[1:]}")
        ax[0].set_xlabel("Time ($s$)")
        ax[1].set_xlabel("Frequency ($Hz$)")
        ax[0].set_ylabel("EEG reading ($\mu V$)")

        # Plot temporal data
        ax[0].plot(native.time, native.get_signal(placement), label="Native Language", linewidth=0.5)
        ax[0].plot(non_native.time, non_native.get_signal(placement), label="Non-Native Language", linewidth=0.5)
        ax[0].plot(neutral.time, neutral.get_signal(placement), label="Natural Music", linewidth=0.5)
        # ax[0].plot(eyes_closed_rest.time, eyes_closed_rest.get_signal(placement), label="Eyes Closed Resting", linewidth=0.5)
        # ax[0].plot(eyes_open_rest.time, eyes_open_rest.get_signal(placement), label="Eyes Open Resting", linewidth=0.5)

        # Plot fourier transforms
        ax[1].plot(native_ft_freq, native_ft, label="Native Language", linewidth=0.5)
        ax[1].plot(non_native_ft_freq, non_native_ft, label="Non-Native Language", linewidth=0.5)
        ax[1].plot(neutral_ft_freq, neutral_ft, label="Natural Music", linewidth=0.5)
        # ax[1].plot(eyes_closed_rest_ft_freq, eyes_closed_rest_ft, label="Eyes Closed Resting", linewidth=0.5)
        # ax[1].plot(eyes_open_rest_ft_freq, eyes_open_rest_ft, label="Eyes Open Resting", linewidth=0.5)

        ax[0].legend(loc="upper right")
        ax[1].legend(loc="upper right")
        plt.show()

        run = False if input("Enter \'q\' to quit, press enter to continue\n\t> ").lower() == "q" else True

if __name__ == "__main__":
    main()