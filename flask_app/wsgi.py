from app import app as application

app = application
print ("wsgi_print_main")
print(__name__)

if __name__ == "__main__":
    print ("wsgi_print_main_1")
    app.run(host='0.0.0.0', port=8001)
    print ("wsgi_print_main_2")

if __name__ == "wsgi":
    print ("wsgi_print_main_3")
    app.run(host='0.0.0.0', port=8001)
    #app.run
    print ("wsgi_print_main_4")

#if __name__ == "__main__":
#    print ("wsgi_print_main_1")
#    app.run(host='0.0.0.0', port=8001)
#    print ("wsgi_print_main_2")
