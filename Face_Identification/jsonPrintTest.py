import time
import json

def main():
    names = {'Akshat Alok', 'Samarth Bhargav', 'Preston Brown', 'Jesse Choe', 'Santiago Criado', 'Tejesh Dandu', 'Shreyan Dey', 'Mihika Dusad', 'Manav Gagvani', 'Om Gole', 'Rohan Kalahasty', 'Darren Kao', 'Dev Kodre', 'Pranav Kuppa', 'Grace Liu', 'Krish Malik', 'Lucas Marschoun', 'Lakshmi Sritan Motati', 'Vishal Nandakumar', 'Matthew Palamarchuk', 'Pranav Panicker', 'Tanvi Pedireddi', 'Ashwin Pulla', 'Daniel Qiu', 'Abhisheik Sharma', 'Ayaan Siddiqui', 'Raghav Sriram', 'Pranav Vadde', 'Akash Wudali'}
    facesDct = {name:False for name in names}
    output_path = '/Users/shreyandey/Documents/Grade12/ML/AttendanceSystem/Face_Identification/identified_people.json'
    faces = {'shreyan_dey','manav_gagvani'}
    for face in faces:
        if not facesDct[formatName(face)]:
            facesDct[formatName(face)] = True
            json_object = json.dumps(facesDct, indent=4)
            with open(output_path, 'w') as f:
                f.write(json_object)
        time.sleep(10)

def formatName(withUnderscore):
    return f"{withUnderscore[0].upper()}{withUnderscore[1:withUnderscore.index('_')]} {withUnderscore[withUnderscore.index('_')+1].upper()}{withUnderscore[withUnderscore.index('_')+2:]}"

if __name__ == "__main__":
    main()