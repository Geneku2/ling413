import sys

orig = open("yelp_academic_dataset_review.json","r", encoding="utf-8")
first_mil = open("yelp_reviews_first_mil.json","w", encoding="utf-8")

for i in range(1000000):
    first_mil.write(orig.readline())