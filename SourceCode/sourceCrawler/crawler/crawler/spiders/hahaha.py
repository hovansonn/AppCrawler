import scrapy
import os

class QuotesSpider(scrapy.Spider):
    name = "hvs" #tên
    domain = '' #tên miền để crawl dữ liệu
    folder_path = 'input/' #đường dẫn file
    page = '' #Tên file chứa dữ liệu crawl
    sub_page = 1
    urlt = []
    namet = []
    index_of_crawl = 0
    selector_of_name = ""
    selector_of_link = ""
    selector_of_sub_name = ""
    def start_requests(self):
        arr = [] #
        file_structure_of_web = open("structure.txt","r")
        for x in file_structure_of_web.readlines():
            x = x.split("\n")[0]
            arr.append(x.split(" "))
        file_structure_of_web.close()
        print(arr)
        loop = 1
        while loop == 1:
            self.domain = input("Nhập địa chỉ trang web bạn muốn thu thập dữ liệu: ") 
                
            if "http" not in self.domain:   
                if "/" in self.domain:
                    self.domain = self.domain.split('/')[0]
                # if "www" not in self.domain:
                #     self.domain = "www." + self.domain
                if "www" in self.domain:
                    self.folder_path = self.folder_path + self.domain.split('.')[1]
                else:
                    self.folder_path = self.folder_path + self.domain.split('.')[0]
            else:
                if "/" in self.domain:
                    self.domain = self.domain.split('/')[2]
                # if "www" not in self.domain:
                #     self.domain = "www." + self.domain
                if "www" in self.domain:
                    self.folder_path = self.folder_path + self.domain.split('.')[1]
                else:
                    self.folder_path = self.folder_path + self.domain.split('.')[0]
            
            
            print(self.domain)
            
            for a in arr:
                if self.domain in a[0]:
                    loop = 2 
                    self.domain = a[0]
                    self.selector_of_name = a[1]
                    self.selector_of_link = a[2]
                    self.selector_of_sub_name = a[1]
                    self.index_of_crawl = int(a[3])
            print(self.selector_of_name)
            print(self.selector_of_link)
            print(self.selector_of_sub_name)
            print(self.index_of_crawl)
            print(self.domain)
            
            if loop ==1:
                print("Trang web này không crawl được vui lòng chọn trang web khác để crawl!!!")
                
        urls =[self.domain] #lấy danh sách các domain cần crawl tiêu đề. trong trường hợp này chỉ có 1 domain do yêu cầu đề bài
        
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse) #duyệt qua list các domain để crawl dữ liệu. Trong trường hợp này ta có thể bỏ luôn vòng for và gọi thẳng lệnh yield hoặc return
            #lệnh yield chức năng giống như return nhưng có một điểm khác là thay vì kết thúc hàm nó sẽ tạm ngưng hàm và khi gọi lại hàm nó sẽ tiếp tục chạy từ khúc có lệnh yield trước đó
            #ví dụ có 2 domain thì đầu tiên nó sẽ request domain thứ 1 rồi lưu lại trạng thái sau khi request domain thứ nhất. 
            # sau đó khi gọi trong list domain vãn còn domain thì nó tiếp tục gọi lại hàm start_request và bắt đầu tại vòng for có domain thứ 2

    def parse(self, response):
        if response.url in self.domain: #kiểm tra xem domain khi request có phải là domain nhập vào hay là domain của tiêu đề được chọn 
            
            title_name = response.css(str(self.selector_of_name)).extract() #lấy danh sách các title của web
            title_link = response.css(str(self.selector_of_link)).extract()  #lấy danh sách cách link đi với các title ở dòng code trên
            title_name_sub = response.css(str(self.selector_of_sub_name)).extract() #lấy các tên của title để làm sub folder
            print("Các chủ đề của trang web %s" % self.domain)
            i = 1
            web_info = [] # biến chứa tất các các danh sách đã lấy ở trên

            for item in zip(title_name,title_link,title_name_sub): #đổ dữ liệu vào biến chứa. Lưu ý(hàm zip cho phép ta gộp các interator lại để duyệt)
                print(str(i)+". "+item[0]) #xuất danh sách các tiêu đề
                if("https" in item[1]):
                    web_info.append([i,item[1],item[2]]) #đổ dữ liệu vào biến chứa
                else:
                    web_info.append([i,self.domain + item[1],item[2]]) #đổ dữ liệu vào biến chứa
                i=i+1

            choose = input("Nhập chủ đề bạn muốn thu thập dữ liệu: ")
            url = ''

            for title in web_info:  #duyệt qua danh sách các tiêu đề để lấy ra tiêu đề được chọn
                if int(choose) == title[0]: #kiểm tra chuỗi
                    url = title[1] #gán url mới bằng url của tiêu đề được chọn
                    print(title[2])
                    if "\n" in title[2]:
                        title[2] = title[2].split("\n")[1]
                    self.folder_path = self.folder_path +"/" + str(title[2]) #gán chuỗi cây thư mục để lúc sau tạo ra cây thư mục
                    self.page = str(title[2]) #lấy đên của tiêu đề để làm tên file chứa dữ liệu
            if("https" in self.folder_path):
                self.folder_path = self.folder_path.split('/')[2]
            if not os.path.exists(self.folder_path): 
                os.makedirs(self.folder_path) 
            return scrapy.Request(url=url,callback = self.parse) #gọi lại hàm request để nhận respone theo tiêu đề
            
        else:
            if self.sub_page == 1:
                self.sub_page = 2
                filename = '_%s.txt' % self.page 
                with open(self.folder_path+"/"+filename, 'wb') as f: 
                    f.write(response.body)
                self.log('Saved file %s' % filename)
                title_name = response.css("a::text").extract() #lấy danh sách các title của web
                title_link = response.css("a::attr(href)").extract()  #lấy danh sách cách link đi với các title ở dòng code trên
                self.urlt = title_link
                self.namet = title_name
                print(self.namet)
                print(self.urlt)
                print(len(self.urlt))
                
                urlrqs = self.urlt[self.index_of_crawl]
                if "http" not in self.urlt[self.index_of_crawl]:
                    urlrqs = self.domain + self.urlt[self.index_of_crawl]
                
                print(urlrqs)
                return scrapy.Request(url=urlrqs,callback=self.parse)
            else:
                
                filename = '_%s.txt' % self.index_of_crawl
                with open(self.folder_path+"/"+filename, 'wb') as f: 
                    f.write(response.body)
                self.log('Saved file %s' % filename)
                self.index_of_crawl = self.index_of_crawl + 1
                
                
                urlrqs = self.urlt[self.index_of_crawl]
                if "http" not in self.urlt[self.index_of_crawl]:
                    urlrqs = self.domain + self.urlt[self.index_of_crawl]

                print(self.index_of_crawl)
                if self.sub_page == 4: 
                    return
                return scrapy.Request(url=urlrqs,callback=self.parse)

            