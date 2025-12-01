# Scripts for Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II)

## Prefill 

![DP attention and EP MoE](./assets/dp-attention-ep-moe.png)

The benchmark is batch size of 1024 and input length of 1024. How to config our params 

For prefill, we use 1 node with 4 GPUs.

With TP=DP=EP=4, and the input request being 1024 requests (one batch), the request division is shown in the graph above. That is, each GPU will receive 1024 / 4 = 256 requests. With the input length of 1024. 

How to set the parameters? 

Important configs are 

--max-total-tokens 128k

--max-prefill-tokens 16k

--chunked-prefill-size 64k

Here each means the total tokens for the prefill pool. So we need to divide the number of GPUs to get per gpu number. 

IF our GPU is capable of handling 16k tokens per forward pass. Since each request is 1k tokens, we will expect 16 requests in one batch. So expect 256/16 requests = 16 forward passes. 

If we want 16k tokens per forward pass, we need to set the --max6 -prefill-tokens per gpu to be at least and --max-total-tokens per gpu at least 16k. It combines prefill and decode tokens, so set this to a large number. chunked-prefill-size per  gpu should also be greater than 16k. Otherwise, your requests will be chopped into muliple forward pass. Note that chunjked prefill size is aggregated, so if you have 4 GPUs, mutiply 16k by 4 to get 64k.

Results (full detail in exp 01)

```
ttft: 17.12 s
input throughput: 61249.67 tok/s
```


Sceniar 2. If found that GPU is not saturated with 16k tokens per forward pass, we can increase it to 32k tokens per forward pass. To do so, we can set 

```
--max-total-tokens 128k

--max-prefill-tokens 32k

--chunked-prefill-size 128k
```


Results is higher throughput (full detail in exp 02)
```
ttft: 16.68 s
input throughput: 62862.29 tok/s
```

Here, max running request wouldn't cap us. Since we usually has few running requests, but many queued requests. We can set this to a big number. 

See launch.sh for full script.

Link to blog: https://lmsys.org/blog/2025-09-25-gb200-part-2/