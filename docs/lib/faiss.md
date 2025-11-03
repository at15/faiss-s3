# Faiss

- [ ] `IndexIVFStats` seems to be a global stats? Seems you can pass it as search parameter
- [ ] Use `search_preassigned` or pass ID selector as parameter to the quantizer?

## SearchParameters

- `inverted_list_context` where is it used? Might be useful for manage caching and record metrics such as cache hits and misses.
  - only used when using iterator with `use_iterator` set to true, by default is false, this is used by the rocksdb demo and `DispatchingInvertedLists`
- `quantizer_params` can we pass ID selector to it? Though we can use `search_preassigned` to specify the cluster id directly

```cpp
struct SearchParametersIVF : SearchParameters {
    size_t nprobe = 1;    ///< number of probes at query time
    size_t max_codes = 0; ///< max nb of codes to visit to do a query
    SearchParameters* quantizer_params = nullptr;
    /// context object to pass to InvertedLists
    void* inverted_list_context = nullptr;

    virtual ~SearchParametersIVF() {}
};
```