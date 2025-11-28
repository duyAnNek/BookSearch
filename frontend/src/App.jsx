import React, { useState } from 'react';
import './App.css';
// SỬA ĐƯỜNG DẪN IMPORT: Thêm .jsx và xóa /components/ (nếu file nằm chung)
// Giả sử bạn đặt components trong thư mục /components:
import SearchBar from './components/SearchBar.jsx';
import ResultsList from './components/ResultsList.jsx';
import BookModal from './components/BookModal.jsx';
import axios from 'axios';

// Cấu hình API endpoint
const API_URL = 'http://localhost:8000';

function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedBook, setSelectedBook] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (query, searchType) => {
    setLoading(true);
    setResults([]);
    setHasSearched(true);
    try {
      let response;
      if (searchType === 'text') {
        response = await axios.get(`${API_URL}/api/search/text`, {
          params: { query: query }
        });
      } else { // searchType === 'image'
        const formData = new FormData();
        formData.append('file', query); // 'query' lúc này là file ảnh
        response = await axios.post(`${API_URL}/api/search/image`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
      }
      setResults(response.data);
    } catch (error) {
      console.error("Lỗi khi tìm kiếm:", error);
      alert("Không thể thực hiện tìm kiếm. Vui lòng kiểm tra console.");
    }
    setLoading(false);
  };

  const handleBookDoubleClick = (book) => {
    setSelectedBook(book);
  };

  const closeModal = () => {
    setSelectedBook(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>📚 Tìm Kiếm Sách Thông Minh</h1>
        <p>Tìm bằng văn bản (tên sách, tác giả, mô tả) hoặc bằng hình ảnh bìa sách</p>
      </header>
      <main>
        <SearchBar onSearch={handleSearch} loading={loading} />
        <ResultsList 
          results={results} 
          onBookSelect={handleBookDoubleClick} 
          apiUrl={API_URL}
          hasSearched={hasSearched}
        />
      </main>
      <footer className="App-footer">
        <p>© {new Date().getFullYear()} Book Image Search. <br />Built by Nguyen Duy An and Nguyen Quoc Huy.</p>
      </footer>
      {selectedBook && (
        <BookModal 
          book={selectedBook} 
          onClose={closeModal} 
          apiUrl={API_URL} 
        />
      )}
    </div>
  );
}

export default App;