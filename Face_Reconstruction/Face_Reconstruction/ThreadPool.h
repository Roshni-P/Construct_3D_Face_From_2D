#pragma once
/// @file ThreadPool.h
/// @brief Thread pool to queue tasks in each thread
/// @namespace std

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
			-> std::future<typename std::invoke_result<F, Args...>::type>
        {
            using ReturnType =
                typename std::invoke_result<F, Args...>::type;

            auto task = std::make_shared<std::packaged_task<ReturnType()>>(
                std::bind(std::forward<F>(f),
                    std::forward<Args>(args)...)
            );

            std::future<ReturnType> result = task->get_future();

            {
                std::lock_guard<std::mutex> lock(queueMutex);

                if (stopThreads)
                    throw std::runtime_error("Submit on stopped ThreadPool");

                tasks.emplace([task]() { (*task)(); });
            }

            cv.notify_one();

            return result;
        }
private:
	std::mutex queueMutex;
	std::condition_variable cv;
	std::vector<std::thread> worker;
	std::queue<std::function<void()>> tasks;
	bool stopThreads = false;

};