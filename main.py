from agent import start_main_span

if __name__ == "__main__":
    result = start_main_span([{"role": "user", "content": "Which stores did the best in 2021?"}])
    print(result)
