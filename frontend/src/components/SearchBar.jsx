import React, { useState, useRef } from 'react';
import axios from 'axios';

function SearchBar({ onSearch, loading }) {
  const [textQuery, setTextQuery] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [searchType, setSearchType] = useState('text');
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const debounceRef = useRef(null);

  const handleTextSubmit = (e) => {
    e.preventDefault();
    if (textQuery.trim()) {
      onSearch(textQuery, 'text');
      setShowSuggestions(false);
      setHighlightIndex(-1);
    }
  };

  const handleInputChange = async (e) => {
    const value = e.target.value;
    setTextQuery(value);

    if (!value.trim()) {
      setSuggestions([]);
      setShowSuggestions(false);
      setHighlightIndex(-1);
      return;
    }

    // Debounce gọi API Gemini ~350ms sau khi người dùng dừng gõ
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(async () => {
      try {
        const resp = await axios.get('http://localhost:8000/api/autocomplete/gemini', {
          params: { query: value }
        });
        const data = Array.isArray(resp.data) ? resp.data : [];
        setSuggestions(data);
        setShowSuggestions(data.length > 0);
        setHighlightIndex(-1);
      } catch (err) {
        console.error('Lỗi autocomplete:', err);
      }
    }, 350);
  };

  const handleKeyDown = (e) => {
    if (!showSuggestions || suggestions.length === 0) {
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlightIndex((prev) => {
        const next = prev + 1;
        return next >= suggestions.length ? 0 : next;
      });
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightIndex((prev) => {
        const next = prev - 1;
        return next < 0 ? suggestions.length - 1 : next;
      });
    } else if (e.key === 'Enter') {
      if (highlightIndex >= 0 && highlightIndex < suggestions.length) {
        e.preventDefault();
        const selected = suggestions[highlightIndex];
        setTextQuery(selected);
        setShowSuggestions(false);
        setHighlightIndex(-1);
        onSearch(selected, 'text');
      }
    } else if (e.key === 'Tab') {
      // Nhấn Tab để nhận gợi ý (không chuyển focus đi)
      e.preventDefault();
      if (suggestions.length > 0) {
        const index =
          highlightIndex >= 0 && highlightIndex < suggestions.length
            ? highlightIndex
            : 0;
        const selected = suggestions[index];
        setTextQuery(selected);
        setShowSuggestions(false);
        setHighlightIndex(-1);
      }
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
      setHighlightIndex(-1);
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      // Tự động tìm kiếm khi chọn ảnh
      onSearch(file, 'image');
      setSearchType('image');
    }
  };

  return (
    <div className="search-bar-container">
      <div className="tabs">
        <button 
          className={searchType === 'text' ? 'active' : ''}
          onClick={() => setSearchType('text')}
        >
          Tìm bằng Văn bản
        </button>
        <button 
          className={searchType === 'image' ? 'active' : ''}
          onClick={() => setSearchType('image')}
        >
          Tìm bằng Hình ảnh
        </button>
      </div>

      {searchType === 'text' ? (
        <form onSubmit={handleTextSubmit} className="search-form">
          <div className="autocomplete-wrapper">
            <input
              type="text"
              value={textQuery}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Nhập tên sách, tác giả, hoặc mô tả..."
              disabled={loading}
            />
            {showSuggestions && suggestions.length > 0 && (
              <ul className="autocomplete-list">
                {suggestions.map((sug, idx) => (
                  <li
                    key={idx}
                    className={idx === highlightIndex ? 'active' : ''}
                    onClick={() => {
                      setTextQuery(sug);
                      setShowSuggestions(false);
                      setHighlightIndex(-1);
                      onSearch(sug, 'text');
                    }}
                  >
                    {sug}
                  </li>
                ))}
              </ul>
            )}
          </div>
          <button type="submit" disabled={loading}>
            {loading ? 'Đang tìm...' : 'Tìm kiếm'}
          </button>
        </form>
      ) : (
        <div className="image-upload-form">
          <input
            type="file"
            id="imageUpload"
            accept="image/*"
            onChange={handleImageChange}
            disabled={loading}
          />
          <label htmlFor="imageUpload" className="image-upload-label">
            {loading ? 'Đang xử lý...' : (imageFile ? imageFile.name : 'Chọn ảnh bìa sách')}
          </label>
        </div>
      )}
    </div>
  );
}

export default SearchBar;