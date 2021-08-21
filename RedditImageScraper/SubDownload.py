import praw
import requests
import cv2
import numpy as np
import os
import pickle
import sys
import cv2
from psaw import PushshiftAPI
from utils.create_token import create_token

try:
  subreddit_name = str(sys.argv[1])
  POST_SEARCH_AMOUNT = int(sys.argv[2])
except:
  print("Wrong arguments!")

# Create directory if it doesn't exist to save images
def create_folder(image_path):
    CHECK_FOLDER = os.path.isdir(image_path)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(image_path)

# Path to save images
dir_path = os.path.dirname(os.path.realpath(__file__))
# output_images_path = 'C:/Users/Jérôme/Desktop/GANbang/image_scrapping/reddit_images'
output_images_path = 'E:/Projets/GANbang/Data/Reddit_images'
image_path = os.path.join(output_images_path, subreddit_name+"/") 
ignore_path = os.path.join(dir_path, "ignore_images/") 
create_folder(image_path)

# Get token file to log into reddit.
# You must enter your....
# client_id - client secret - user_agent - username password

# if os.path.exists('token.pickle'):
#     with open('token.pickle', 'rb') as token:
#         creds = pickle.load(token)
# else:
#     creds = create_token()
#     pickle_out = open("token.pickle","wb")
#     pickle.dump(creds, pickle_out)

reddit = praw.Reddit(client_id='CdI1ICLA5Z0NqctxTdJGTQ',
                    client_secret='axg7mXgwij8f-sVu--5dO2_PiLALEw',
                    user_agent='ImageScraper',
                    username='',
                    password='')


# f_final = open("sub_list.csv", "r")
img_notfound = cv2.imread('imageNF.png')

sub = subreddit_name
subreddit = reddit.subreddit(sub)
face_cascade = cv2.CascadeClassifier('RedditImageScraper/utils/haarcascade_frontalface_default.xml')
upperbody_cascade = cv2.CascadeClassifier('RedditImageScraper/utils/haarcascade_upperbody.xml')
fullbody_cascade = cv2.CascadeClassifier('RedditImageScraper/utils/haarcascade_fullbody.xml')

print(f"Starting parsing images from r/{sub}!")
print("Saving folder :", output_images_path, "\n")
count_valid = 0

api = PushshiftAPI(reddit)
#sort_type='score', sort='desc'

for submission in  api.search_submissions(subreddit=subreddit_name, limit=100000) :#subreddit.top(limit=1000000):
  if "jpg" in submission.url.lower() or "png" in submission.url.lower():
      try:
          resp = requests.get(submission.url.lower(), stream=True).raw
          image = np.asarray(bytearray(resp.read()), dtype="uint8")
          image = cv2.imdecode(image, cv2.IMREAD_COLOR)

          # Could do transforms on images like resize!
          compare_image = cv2.resize(image,(224,224))

          # Get all images to ignore
          for (dirpath, dirnames, filenames) in os.walk(ignore_path):
              ignore_paths = [os.path.join(dirpath, file) for file in filenames]
          ignore_flag = False

          for ignore in ignore_paths:
              ignore = cv2.imread(ignore)
              difference = cv2.subtract(ignore, compare_image)
              b, g, r = cv2.split(difference)
              total_difference = cv2.countNonZero(b) + cv2.countNonZero(g) + cv2.countNonZero(r)
              if total_difference == 0:
                  ignore_flag = True

          if not ignore_flag:
              #Face detection filter
              gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
              faces = np.array(face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=8,minSize=(300,300))) 
              # upper_body = np.array(upperbody_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5,minSize=(300,300)))
              #full_body = np.array(fullbody_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5,minSize=(300,300)))

              if faces.any(): #or upper_body.any() or full_body.any():
                print(f"{count_valid+1} valid image saved")
                # for (x, y, w, h) in faces:
                #     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imwrite(f"{image_path}{sub}-{submission.id}.png", image)
                count_valid += 1

      except Exception as e:
          print(f"Image failed. {submission.url.lower()}")
          print(e)

      if count_valid >= POST_SEARCH_AMOUNT :
        print("finished")
        break
        