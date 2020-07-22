import os
import re
from bs4 import BeautifulSoup
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy import spatial
from gensim.summarization.bm25 import get_bm25_weights
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

#lấy danh sách các stopword
my_stopwords = set(stopwords.words('english') + list(punctuation))
#Lấy dữ liêu từ file
def get_text(file):
    read_file = open(file,'r', encoding="utf8")
    text = read_file.readlines()
    text = ' '.join(text)
    return text

#Xóa các thẻ html trong file
def clean_html(text):
    soup = BeautifulSoup(text,'html.parser')
    return soup.get_text()
#xoát các ký tự đặc biệt
def remove_special_character(text):
    string = re.sub('[^\w\s]','',text)
    string = re.sub('\s+',' ',string)
    string = string.strip()
    return string
#Đếm tần suất xuất hiện mỗi từ
def CountFrequency(arr):
    return Counter(arr)


if __name__ == "__main__":
    #Duyệt và lấy danh sách file thuộc thư mục đầu vào của chương trình(file_input)
    path_input = input("Nhập thư mục đầu vào: ")
    path_output = input("Nhập thư mục đầu ra: ")
    choose_method = input("Chọn phương pháp tính toán vector(1.Bag of Word, 2.TF-IDF): ")
    choose_relevance = input("Chọn phương pháp tính độ tương đồng(1. CosSim, 2. Okapi BM25): ")

    #nếu folder chưa tồn tại thì tạo folder mới
    if os.path.isdir(path_output) == False:
        os.mkdir(path_output)
        

    arr_TFIDF = []
    list_path = []
    list_name = []
    list_document_after_preprocess= []
    list_word = []
    for root, dirs, files, in os.walk(path_input):
        for file in files :
            list_path.append(root+"/"+file)
            list_name.append(os.path.splitext(file)[0])

    file_summary = open(path_output + "/file_summary.txt", "w")

    #Tiền xử lý dữ liệu
    for i in range(len(list_path)):
        #đọc dữ liệu, xóa thẻ html và ghi ra file
        text = get_text(list_path[i])

        text_cleaned = clean_html(text)
        
        file_after_remove_html_tag = open(path_output+'/file_after_remove_html_tag_'+str(i)+'.txt',"w", encoding="utf8")
        file_after_remove_html_tag.write(text_cleaned)
        file_after_remove_html_tag.close()

        #tách câu và ghi ra file
        sents = sent_tokenize(text_cleaned) 
        file_after_separate_the_sentence = open(path_output+'/file_after_separate_the_sentence_'+str(i)+'.txt',"w", encoding="utf8")
        #convert list sents sang chuỗi
        str_sents = ''.join(sents)
        file_after_separate_the_sentence.write(str_sents)
        file_after_separate_the_sentence.close()

        #xóa các ký tự đặc biệt và ghi ra file
        sents_cleaned = [remove_special_character(s) for s in sents]
        
        file_after_remove_special_character_in_sentence = open(path_output+'/file_after_remove_special_character_in_sentence_'+str(i)+'.txt',"w", encoding="utf8")
        #convert list sents_cleaned sang chuỗi
        str_sents_cleaned = ''.join(sents_cleaned)
        file_after_remove_special_character_in_sentence.write(str_sents_cleaned)
        file_after_remove_special_character_in_sentence.close()
        #nối các câu lại
        text_sents_join = ' '.join(sents_cleaned)
        #tách từ
        words = word_tokenize(text_sents_join)
        
        #đếm tần suất xuất hiện mỗi từ
        arr_word = CountFrequency(words)
        #ghi ra file
        file_after_separate_the_word = open(path_output+'/file_after_separate_the_word_'+str(i)+'.txt',"w", encoding="utf8")
        for (word,fre) in arr_word.items():
            file_after_separate_the_word.write(word+ ':'+ str(fre) +'\n')
        file_after_separate_the_word.close()
        #đưa về dạng chữ thường
        words = [word.lower() for word in words]
        #Loại bỏ hư từ và ghi ra file
        words = [word for word in words if word not in my_stopwords]

        arr_word = CountFrequency(words)

        file_after_remove_stopword = open(path_output+'/file_after_remove_stopword_'+str(i)+'.txt',"w", encoding="utf8")
        for (word,fre) in arr_word.items():
            file_after_remove_stopword.write(word+ ':'+ str(fre) + '\n')
        file_after_remove_stopword.close()
        #chuẩn hóa từ
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words]
        list_word.append(words)

        #tạo mảng các document sau khi chuẩn hóa
        str_words = ' '.join(words)
        list_document_after_preprocess.append(str_words)
        

        #ghi file tổng kết
        len_words = len(words)
        file_summary.write( list_name[i] + ": " + str(len(sents_cleaned)) + ", " + str(len_words) + '\n')
        
        arr_word = CountFrequency(words)
        file_final = open(path_output+'/'+list_name[i]+'_word.txt',"w", encoding="utf8")
        #ghi các từ và tầng số vào file
        for (word,fre) in arr_word.items():
            file_final.write(word+ ':'+ str(fre) + '\n')
        file_final.close()

    #---------------------------------------------------------------------------------------------------------------
    list_result_after_convert_vector = []
    str_method = ""
    #Bag of word
    if choose_method ==  '1':
        BoW_result = CountVectorizer()
        BoW_result_matrix = BoW_result.fit_transform(list_document_after_preprocess).todense()
        # print(BoW_result.transform(list_document_after_preprocess))
        # print(len(BoW_result.get_feature_names()))
        BoW_list_result = BoW_result_matrix.tolist()

        list_result_after_convert_vector = BoW_list_result
        str_method = "BoW"

        file_BoW = open(path_output+'/' + 'BoW.txt',"w", encoding="utf8")

        for i in range(len(BoW_list_result)):  
            str_result = ' '.join(str(e) for e in BoW_list_result[i])
            file_BoW.write(str(i + 1) + " " + list_name[i] + ".txt " + str_result + '\n')

        file_BoW.close()
    elif choose_method == '2':
    #TF-IDF
        tf = TfidfVectorizer(analyzer = 'word', ngram_range=(1, 3), min_df=0, stop_words='english')
        tf_idf_matrix = tf.fit_transform(list_document_after_preprocess)
        # print(tf_idf_matrix) ma tra tf-idf của từng từ gòm (document, vị trí)
        feature_names = tf.get_feature_names()

        dense = tf_idf_matrix.todense()

        TFIDF_list_result = dense.tolist()

        arr_TFIDF = np.array(TFIDF_list_result)


        list_result_after_convert_vector = TFIDF_list_result
        str_method = "TF-IDF"

        file_TFIDF = open(path_output+'/' + 'TF-IDF.txt',"w", encoding="utf8")
        for i in range(len(TFIDF_list_result)):  
            str_result = ' '.join(str(round(e, 3)).ljust(5) for e in TFIDF_list_result[i])
            file_TFIDF.write(str(i + 1) + " " + list_name[i] + ".txt " + str_result + '\n')
        
        file_TFIDF.close()

        #knn phân lớp cho tài liệu

        #gán nhãn cho từng tài liệu
        lable = [0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 
            1, 1 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        arr_lable = np.array(lable)

        X = arr_TFIDF
        y = arr_lable

        #chia dữ liệu train và dữ liệu test
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

        #huấn luyện cho mô hình knn bằng dữ liệu train
        knn = KNeighborsClassifier(n_neighbors = 5)

        classifier = knn.fit(X_train, y_train)
    
        #truy vẫn tại liệu liên quan
        print("truy vấn 5 tài liệu liên quan đến từng tài liệu trong dữ liệu X_test: ")
        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(X_train)
        (distance, found_index) = neigh.kneighbors(X_test)
        
        for i in range(len(found_index)):
            print("tài liệu liên quan đến X_test: ", i)
            print(found_index[i], "khoảng cách tương ứng", distance[i])

        
        #dự đoán dựa trên dữ liệu test
        y_pred = knn.predict(X_test)

        #đánh giá kết quả
        accuracy = knn.score(X_test, y_test)
        precision = precision_score(y_test, y_pred, average=None)
        recall = recall_score(y_test, y_pred, average=None)
        f_score = f1_score(y_test, y_pred, average=None)

        print("Đánh giá hệ thống knn")
        print("accuracy: ", accuracy)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f_score: ", f_score)

        # vẽ ma trận nhầm lẫn
        class_names = np.array(['health', 'tech', 'business'])
        np.set_printoptions(precision=2)

        titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

        for title, normalize in titles_options:
            disp = plot_confusion_matrix(classifier, X_test, y_test,
                                        display_labels=class_names,
                                        cmap=plt.cm.Blues,
                                        normalize=normalize)
            disp.ax_.set_title(title)

            print(title)
            print(disp.confusion_matrix)

        plt.show()

            
        
    #---------------------------------------------------------------------------------------------------------------
    #Tính độ tương đồng

    #Phương pháp CosSim
    if choose_relevance ==  '1':  
        file_CosSim = open(path_output+'/' + str_method + '_CosSim.txt',"w", encoding="utf8")
        for item1 in list_result_after_convert_vector:
            str_result = ""

            for item2 in list_result_after_convert_vector:
                result = 1 - spatial.distance.cosine(item1, item2)
                result = round(result, 5)
                str_result += str(result).ljust(10)

            file_CosSim.write(str_result + '\n')

        file_CosSim.close()
    #phương pháp Okapi BM25
    elif choose_relevance == '2':
        result_bm25 = get_bm25_weights(list_word, n_jobs=-1)

        file_BM25 = open(path_output+'/' + str_method + '_OkapiBM25.txt',"w", encoding="utf8")

        for item in result_bm25:
            str_result = ' '.join(str(round(e, 3)).ljust(10) for e in item)
            
            file_BM25.write(str_result + '\n')

        file_BM25.close()

    