from app import server

server.config['DEBUG'] = False
print (f"wsgi_print_main:{__name__}")

if __name__ == "__main__":
    print (f"Running Server from:{__name__}")
    server.run(host='0.0.0.0', port=8001)

if __name__ == "wsgi":
    print (f"Running Server from:{__name__}")
    server.run(host='0.0.0.0', port=8001)