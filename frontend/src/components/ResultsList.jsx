import React from 'react';

function ResultsList({ results, onBookSelect, apiUrl, hasSearched }) {
  if (results.length === 0) {
    // Chỉ hiển thị thông báo sau khi người dùng đã thực hiện ít nhất 1 lần tìm kiếm
    return hasSearched ? (
      <p className="no-results">Không tìm thấy kết quả nào.</p>
    ) : null;
  }

  // Hàm tạo URL đầy đủ cho ảnh
  const getImageUrl = (imagePath) => {
    // imagePath lưu trong DB là 'all_covers/image.jpg'
    // Cần request /images/all_covers/image.jpg
    return `${apiUrl}/images/${imagePath}`;
  };

  return (
    <div className="results-grid">
      {results.map((book) => (
        <div 
          key={book.id} 
          className="result-item" 
          onDoubleClick={() => onBookSelect(book)}
          title="Nhấp đúp để xem chi tiết"
        >
          <img 
            src={getImageUrl(book.image_path)} 
            alt={book.title} 
            onError={(e) => { e.target.src = 'https://via.placeholder.com/150?text=No+Image'; }} // Ảnh dự phòng
          />
          <p>{book.title}</p>
        </div>
      ))}
    </div>
  );
}

export default ResultsList;