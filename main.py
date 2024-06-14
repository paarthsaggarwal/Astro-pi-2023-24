# Importing relevant modules
from datetime import datetime, timedelta
import cv2 # importing opencv module to be used in identifying same features in images taken by the PiCamera
import math
from picamera import PiCamera # importing picamera to take images which will be analysed by opencv library for similar features
from pathlib import Path
from orbit import ISS
from time import sleep
import csv
import numpy as np

################################# OUR APPROACH ###################################### 
'''
1. Capturing 30 images (1 image every 5 seconds) uisng PICamera, e.g. image_00001 to image_00030
2. It takes time to store the image on the disk, so we capture the time when the image is stored on the disk in file 01_image_creation_time.txt
3. We identified similar features in images using the cv2 library and the ORB model, calculated the pixel distance between these features, and then the real distance
4. We filter out the outliers for the time values obtained in step 2 and calculate the mean of the remaining values 
6. The ISS speed is calculated between the 2 images and is stored on the disk in file 01_intermediate_speed_data.txt and repeat this process for all 36 images
7. We filter out the outliers for the speed values and calculate the mean of the remaining values 
8. We round this speed value to 5 significant figures
'''
########################################################################################


time_difference = 5 # to make sure that each image is taken after 5 seconds
count_limit = 30 # maximum number of photos taken to stay within 42 images limit
base_folder = Path(__file__).parent.resolve()
image_creation_time = base_folder / '01_image_creation_time.txt' # creating a .txt file to store time values
image_creation_time_mean = base_folder / '02_image_creation_time_mean.txt' # creating a .txt file to store mean time of time values in 01_image_creation_time.txt 
intermediate_speed_data = base_folder / '03_intermediate_speed_data.txt' # creating a .txt file to store speed of ISS between images
result = base_folder / 'result.txt' # creating a .txt file thaat will serve as our final answer for the speed of the ISS

# Deleting all contents of the .txt files at the start of the program
with open(image_creation_time, 'w') as clear_file:
    clear_file.truncate(0)

with open(image_creation_time_mean, 'w') as clear_file:
    clear_file.truncate(0)

with open(intermediate_speed_data, 'w') as clear_file:
    clear_file.truncate(0)

with open(result, 'w') as clear_file:
    clear_file.truncate(0)

# Setting the resolution of the camera as (4056, 3040)
cam = PiCamera()
cam.resolution = (4056, 3040)

#Function to convert images to a format which the CV2 library can 'read' the image
def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv

#Function to identify features in the image 
def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

#Function to identify similar features in 2 images 
def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

#Function to calculate the coordinates of the specific features in images 
def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2

#Function to calculate the distance between the similar features in images
def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    return all_distances / len(merged_coordinates)

#Function to calculate the speed im km/s using the distance value obtained from the "calculate_mean_distance" function
def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed

counter = 0 #Defining a counter variable
count = 0 #Defining another counter variable

start_time = datetime.now()
now_time = datetime.now()


#Ensuring while loop will break after 5 mins to keep inside time limit
while (now_time < start_time + timedelta(minutes=5)): 
    try:
        if count == count_limit or count > count_limit: #Ensuring while loop breaks after capturing all images needed 
            break
        else:
            count += 1
            now_time = datetime.now()
            first_time = now_time

            image_1 = cam.capture(f"{base_folder}/image_{count:05d}.jpg")
            now_time = datetime.now()
            second_time = now_time
            # print("Image: ", f"{base_folder}/image_{count:05d}.jpg", " stored at: ", second_time)

            sleep(time_difference)
            
            count += 1
            now_time = datetime.now()
            third_time = now_time
            image_2 = cam.capture(f"{base_folder}/image_{count:05d}.jpg")
            now_time = datetime.now()
            fourth_time = now_time
            # print("Image: ", f"{base_folder}/image_{count:05d}.jpg", " stored at: ", fourth_time)

            now_time = datetime.now()

            time_difference_1_3 = third_time - first_time
            time_difference_2_4 = fourth_time - second_time

            timestamp_str1 = second_time.strftime("%Y-%m-%d %H:%M:%S.%f")
            timestamp_str2 = fourth_time.strftime("%Y-%m-%d %H:%M:%S.%f")


            # Convert the timestamp strings to datetime objects
            timestamp1 = datetime.strptime(timestamp_str1, "%Y-%m-%d %H:%M:%S.%f")
            timestamp2 = datetime.strptime(timestamp_str2, "%Y-%m-%d %H:%M:%S.%f")

            # Calculate the time difference in seconds
            time_diff_seconds = (timestamp2 - timestamp1).total_seconds()
            
            #calculating the new time difference which includes the time it takes for the image to be saved to the disk
            new_time_difference = time_diff_seconds

            data_row0 = [time_diff_seconds]
            with open(image_creation_time, 'a') as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow(data_row0)
            sleep(time_difference)

    except Exception as e:
        fan = 1

# Read time values between consecutive images when they are stored on the disk
with open(image_creation_time, 'r') as file:
    values_0 = [float(line.strip()) for line in file.readlines()]

# Calculate the interquartile range (IQR) of the time values above
Q1 = np.percentile(values_0, 25)
Q3 = np.percentile(values_0, 75)
IQR = Q3 - Q1

# Define boundaries for outliers
lower_bound_0 = Q1 - 1.5 * IQR
upper_bound_0 = Q3 + 1.5 * IQR

# Filter out outliers
filtered_values_0 = [val for val in values_0 if val >= lower_bound_0 and val <= upper_bound_0]

# Calculate the mean of values without outliers
time_difference_0 = np.mean(filtered_values_0)

# Writing the mean time value of all image storage to the file
mean_time_difference = [time_difference_0]
with open(image_creation_time_mean, 'a') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(mean_time_difference) 

# print("The mean time to store images (after removing outliers): ", time_difference_0)

now_time = datetime.now()
start_time = datetime.now()

count_limit = count

counter = 0 #Defining a counter variable
count = 1 #Defining another counter variable

'''
Here we are evaluating the distance travelled by ISS using the ORB ML model 
and using the time value obtained from earlier (stored in the the variable time_difference_0)
'''
while (now_time < start_time + timedelta(minutes=4)): #Ensuring while loop breaks after 4 mins to keep inside time limit
    try:
        '''
        Ensuring program will stop after all images taken have been 
        taken into account and calculated for the speed
        '''
        if count == count_limit or count > count_limit: #Ensuring while loop breaks after analysing all images taken
            break
        else:
            counter += 1
            now_time = datetime.now()
            image_1 = (f"{base_folder}/image_{counter:05d}.jpg")
            
            count += 1
            image_2 = (f"{base_folder}/image_{count:05d}.jpg")
            now_time = datetime.now()

            # print("Comparing: ", image_1, " with ", image_2)
            # print("Time difference between images: ", time_difference_0)

            image_1 = image_1
            image_2 = image_2
            image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # Create OpenCV image objects
            keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) # Get keypoints and descriptors
            matches = calculate_matches(descriptors_1, descriptors_2) # Match descriptors
            #display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) # Display matches
            coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
            average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)

            speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference_0)
            now_time = datetime.now()

            # print("ISS Speed between : ", image_1, " and ", image_2, " is: ", speed)

            data_row = [str(speed)]
            #Saving each speed value into the relevant .txt file    

            with open(intermediate_speed_data, 'a') as file: 
                writer = csv.writer(file, delimiter='\t')
                writer.writerow(data_row)
            
            now_time = datetime.now()

    except Exception as e:
        fan = 1


# Read values from the .txt file which contains the ISS speed values between images
with open(intermediate_speed_data, 'r') as file:
    values = [float(line.strip()) for line in file.readlines()]

# Calculate the interquartile range (IQR) of the ISS speed values stored above
Q1 = np.percentile(values, 25)
Q3 = np.percentile(values, 75)
IQR = Q3 - Q1

# Define boundaries for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
filtered_values = [val for val in values if val >= lower_bound and val <= upper_bound]

# Calculate the mean of speed values without outliers
mean_without_outliers = np.mean(filtered_values)

# print(f"Average ISS Speed (after removing outliers) is : {mean_without_outliers}")

estimate_kmps = mean_without_outliers  # Replacing with the estimate

# Format the estimate_kmps to have a precision of 5 significant figures

if estimate_kmps < 10: 
    estimate_kmps_formatted = "{:.4f}".format(estimate_kmps)

    # Create a string to write to the file
    output_string = estimate_kmps_formatted

    # Write to the file
    with open(result, 'w') as file:
        file.write(output_string)

    # print("ISS speed written into", result)

elif estimate_kmps >= 10: 
    estimate_kmps_formatted = "{:.3f}".format(estimate_kmps)

    # Create a string to write to the file
    output_string = estimate_kmps_formatted

    # Write to the file
    with open(result, 'w') as file:
        file.write(output_string)

    # print("ISS speed written into", result)

