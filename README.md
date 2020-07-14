# neuromatch-academy-matching
Matching algorithm for Neuromatch Academy 2020

 
## Usage
Prepare `pod.map.csv`, `student.abstracts.csv`, `mentor.info.xlsx` and place them in the root folder. Run `main.py` to generate the matches.

Pod-mentor matching for week 1 will be exported to `pod.schedule.csv` and `mentor.schedule.csv`. For each pod, one half-hour slot in the project time before the fourth or fifith core session will be assigned with a mentor. Affinity based on abstract topic or dataset compatibility is used to find better matches.
