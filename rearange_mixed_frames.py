import os
import sys
from pathlib import Path
import argparse
import cv2
from PIL import Image
import numpy as np
from collections import OrderedDict

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

os.chdir(sys.path[0])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Reorder the mixed frames of a video given, and remove noisy frames"
    )

    parser.add_argument("--video-path", type=Path, help="path to video", required=True)

    # parser.add_argument("--arch", type=str, default="resnet18")

    parser.add_argument("--cluster-method", type=str, help="clustering algorithm used ; choose between kmeans, GMM, hierarchical", default="hierarchical")

    parser.add_argument("--apply-pca", type=int, help="if this arg is present, applies PCA to lower the dimensionality of the embeddings before clustering")

    parser.add_argument("--outliers-gt-file", type=str, help="path to file containing the ground truth labels for the outlier frames", default="./ground_truth_labels.txt")

    parser.add_argument("--reverse-order", "-r", action='store_true', help="if this arg is present, the frames will be ordered in reverse order")

    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    return parser



def get_resnet_embeddings_of_video(feature_extractor, video_path):
    outputs = None
    tensor_frames = []

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #extract frames from video
    vidcap = cv2.VideoCapture(str(video_path))
    success, frame = vidcap.read()
    if(not success):
        raise ValueError("Video not found")
    while (success):
        #preprocess the frames so that we can later feed them to resnet18
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        tensor_frames.append(frame)
        success, frame = vidcap.read()
    vidcap.release()
    print(f"Number of frames: {len(tensor_frames)}")

    #stack in a mini batch
    tensor_frames = torch.stack(tensor_frames)
    tensor_frames = tensor_frames.to(device)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    with torch.no_grad():
        outputs = feature_extractor(tensor_frames)
    return outputs.squeeze()


def resize_image(image, max_dimension_cap=500):
    img_height, img_width, _ = image.shape
    largest_dimension = max(img_height, img_width)
    resized_img = image
    if(largest_dimension > max_dimension_cap):
        resized_img = cv2.resize(image, (int(img_width * max_dimension_cap / largest_dimension), int(img_height * max_dimension_cap / largest_dimension)))
    return resized_img


def visualise_clustering_results(video_path, clustering_results):
    #extract frames from video
    frames = []
    vidcap = cv2.VideoCapture(str(video_path))
    success, frame = vidcap.read()
    if(not success):
        raise ValueError("Video not found")
    while (success):
        #resize the images
        frame = resize_image(frame, max_dimension_cap=1000)
        frames.append(frame)
        success, frame = vidcap.read()
    vidcap.release()
    
    for i, (frame, cluster) in enumerate(zip(frames, clustering_results)):
        cv2.imshow("Frame", frame)
        print(f"Frame {i} assigned to cluster {cluster}")
        key = cv2.waitKey(0)
        if key == ord('q'):#quit
            break
    cv2.destroyAllWindows()


def outliner_filter_accuracy(clustering_results, ground_truth_labels_filename):
    result = (None, None)
    if(os.path.isfile(ground_truth_labels_filename)):
        #get content of ground truth file
        with open(ground_truth_labels_filename, "r") as f:
            ground_truth_labels = f.readlines()
        ground_truth_labels = [int(label) for label in ground_truth_labels]
        correctly_classified = clustering_results == ground_truth_labels
        missclassified_indices = np.where(correctly_classified == False)[0]
        result = np.sum(clustering_results == ground_truth_labels) / len(ground_truth_labels), missclassified_indices
    else:
        print("Ground truth labels for outlier frames file not found")
    return result


def sqdist(vector):
    return sum(x*x for x in vector[1])

#will order frames by putting frames will lowest pairwise similarity next to each other
def sort_frames(embeddings_dict, reverse_order=False):
    sorted_frames_indices = []
    unsorted_frames_indices = list(embeddings_dict.keys())
    sorted_frames_indices.append(unsorted_frames_indices.pop(0))#we place one frame as a starting point
    while (len(unsorted_frames_indices) > 0):
        #find the frame with the highest similarity to the last frame in the sorted list
        max_similarity = -1
        max_similarity_index = -1
        insert_at_end = False
        for idx in unsorted_frames_indices:
            for j in [0, -1]:
                similarity = cosine_similarity([embeddings_dict[sorted_frames_indices[j]]], [embeddings_dict[idx]])[0][0]
                if(similarity > max_similarity):
                    max_similarity = similarity
                    max_similarity_index = idx
                    if(j == 0):
                        insert_at_end = False
                    else:
                        insert_at_end = True
        if(insert_at_end):
            sorted_frames_indices.append(max_similarity_index)
        else:
            sorted_frames_indices.insert(0, max_similarity_index)
        unsorted_frames_indices.remove(max_similarity_index)
    if(reverse_order):
        sorted_frames_indices.reverse()
    return sorted_frames_indices



def recreate_video(video_path, sorted_frames_indices, reversed_order=False):
    #extract frames from video
    frames = []
    vidcap = cv2.VideoCapture(str(video_path))
    original_fps = vidcap.get(cv2.CAP_PROP_FPS)

    success, frame = vidcap.read()
    if(not success):
        raise ValueError("Video not found")
    while (success):
        #resize the images
        frame = resize_image(frame, max_dimension_cap=1000)
        frames.append(frame)
        success, frame = vidcap.read()
    vidcap.release()

    #recreate the video in different order
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    recreated_video_filename = f'{os.path.dirname(os.path.normpath(video_path))}{os.path.basename(os.path.normpath(video_path)).split(".")[0]}_recreated'
    if(reversed_order):
        recreated_video_filename += "_reversed"
    recreated_video_filename += ".mp4"
    out = cv2.VideoWriter(recreated_video_filename, fourcc, original_fps, (frames[0].shape[1], frames[0].shape[0]))
    for i in sorted_frames_indices:
        out.write(frames[i])
    out.release()
    print(f"Video recreated and saved as {recreated_video_filename}")


def main(args):
    #first step: get resnet embeddings of all images in the video, to encode the images' informations into a lower-dimensional space (for resnet18, it's 512 dimensions)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]) #remove the last layer (fc layer) to get the embeddings and not the cls logits
    embeddings =  get_resnet_embeddings_of_video(feature_extractor, args.video_path)

    print(f"Embeddings shape: {embeddings.shape}") #should be (num_frames, 512)

    #second step: clusterise the embeddings using kmeans and filter outliers
    #convert to numpy to perform clustering
    embeddings = embeddings.cpu().numpy()
    embeddings_pca = None
    clustering_results = None
    if(args.apply_pca):
        pca = PCA(n_components=args.apply_pca)
        embeddings_pca = pca.fit_transform(embeddings)
        print(f"Embeddings shape after PCA dim reduction: {embeddings_pca.shape}") #should be (num_frames, args.apply_pca)
    
    if(args.cluster_method.lower() == "kmeans"):
        num_clusters = 2
        kmean_model = KMeans(num_clusters, n_init='auto', random_state=0)
        if(args.apply_pca):
            clustering_results = kmean_model.fit_predict(embeddings_pca)
        else:
            clustering_results = kmean_model.fit_predict(embeddings)
    elif(args.cluster_method.lower() == "gmm"):
        gmm = GaussianMixture(n_components=1, n_init=10, random_state=10)
        densities = None
        if(args.apply_pca):
            gmm.fit(embeddings_pca)
            densities = gmm.score_samples(embeddings_pca)
        else:
            gmm.fit(embeddings)
            densities = gmm.score_samples(embeddings)
        density_threshold = np.percentile(densities, 10)
        print(f"Density threshold: {density_threshold}")
        clustering_results = np.array([densities < density_threshold]).astype(int).squeeze()
        print(f"Cluster results :\n{clustering_results}")
    elif(args.cluster_method.lower() == "hierarchical"):
        hc = AgglomerativeClustering(n_clusters = 2)
        if(args.apply_pca):
            clustering_results = hc.fit_predict(embeddings_pca)
        else:
            clustering_results = hc.fit_predict(embeddings)
    else:
        raise ValueError("Unknown clustering method")
    
    unique, counts = np.unique(clustering_results, return_counts=True)
    print("Clustering cluster sizes:")
    print(dict(zip(unique, counts)))
    main_cluster = unique[np.argmax(counts)]
    print(f"Main cluster: {main_cluster}")

    #check clustering results
    clustering_acc, missclassified_indices = outliner_filter_accuracy(clustering_results, args.outliers_gt_file)
    if(clustering_acc):
        print(f"Clustering accuracy: {clustering_acc}")
        print(f"Missclassified indices: {missclassified_indices}")
    # visualise_clustering_results(args.video_path, clustering_results)

    #get the indices of the frames that are in the main cluster, to remove outlier frames
    main_cluster_indices = np.where(clustering_results == main_cluster)[0]
    #orderedDict with key = frame index, value = embedding
    embeddings_dict = OrderedDict()
    for i in main_cluster_indices:
        embeddings_dict[i] = embeddings[i]

    sorted_frames_idx = sort_frames(embeddings_dict, args.reverse_order)
    recreate_video(args.video_path, sorted_frames_idx, args.reverse_order)
    print("Done!")




if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    main(args)