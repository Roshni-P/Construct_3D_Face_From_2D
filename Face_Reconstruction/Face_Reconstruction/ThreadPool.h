#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <future>

class ThreadPool
{
public:
	ThreadPool(size_t numThreads);
	~ThreadPool();
	template <class F, class... Args>
		auto ProcessThread(F&& f, Args&&... args) 
			-> std::future<typename std::invoke_result<F, Args...>::type>;
private:
	std::mutex queueMutex;
	std::condition_variable cv;
	std::vector<std::thread> worker;
	std::queue<std::function<void()>> tasks;
	bool stopThreads = false;

};