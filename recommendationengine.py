
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

aparc1

# Sample content: articles, blogs, etc.
documents = [
    "Machine learning is a method of data analysis that automates analytical model building.",
    "Deep learning, a subset of machine learning, utilizes neural networks in learning from data.",
    "AI is an interdisciplinary science with multiple approaches, but advancements in machine learning are causing a paradigm shift.",
    "Big data is a term that describes the large volume of data – both structured and unstructured.",
    "Cloud computing is the delivery of computing services—including servers, storage, databases, networking, over the cloud."
]

# Compute the TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Function to get recommendations
def get_recommendations(index, tfidf_matrix):
    # Compute the cosine similarity of the chosen content with others
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    
    # Get top 2 similar content indices
    related_docs_indices = cosine_similarities.argsort()[:-3:-1]
    
    return [documents[i] for i in related_docs_indices[1:]]

# Get recommendations for the first document
index_to_recommend = 0
recommendations = get_recommendations(index_to_recommend, tfidf_matrix)
print(f"Selected content: \n{documents[index_to_recommend]}\n")
print("Recommended content:")
for rec in recommendations:
    print(f"- {rec}")











































# Given an integer list-nums, partition it into two (contiguous) sublists left and right so that:

# 1. Every element in left is less than or equal to every element in right.
# 2. left and right are non-empty.
# 3. left has the smallest possible size.

##Output:
# Return the length of left after such a partitioning.

##Constraints:
# 2 <= nums.length <= 10^5
# 0 <= nums[i] <= 10^6
# There is at least one valid answer for the given input.

##Example 1:
#Input: nums = [5,0,3,5,8,6]
#Output: 3
#xplanation: left=[5,0,3], right=[5,8,6]

##Example 2:
#Input: nums = [1,1,1,0,6,12]
#Output: 4
#Explanation: left = [1,1,1,0], right = [6,12]


def partition_list(nums):
    if not nums:
        return 0
    max_val = nums[0]
    partition_idx = 0
    
    for i in range(1, len(nums)):
        if nums[i] > max_val:
            partition_idx = i - 1
            max_val = nums[i]
            
    left = nums[:partition_idx]
    print(left)
    return len(left)

nums = [1,1,1,0,6,12]
length_of_left = partition_list(nums)
print(length_of_left)
            



# def part_len():
    


