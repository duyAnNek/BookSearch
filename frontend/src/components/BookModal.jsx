import React from 'react';

function BookModal({ book, onClose, apiUrl }) {

  // Xử lý sự kiện click nền để đóng modal
  const handleBackdropClick = (e) => {
    if (e.target.className === 'modal-backdrop') {
      onClose();
    }
  };

  // Hàm tạo URL đầy đủ cho ảnh
  const getImageUrl = (imagePath) => {
    return `${apiUrl}/images/${imagePath}`;
  };

  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div className="modal-content">
        <button className="modal-close-button" onClick={onClose}>X</button>
        
        <div className="modal-body">
          <div className="modal-image-container">
            <img 
              src={getImageUrl(book.image_path)} 
              alt={book.title}
              onError={(e) => { e.target.src = 'https://via.placeholder.com/300?text=No+Image'; }}
            />
          </div>
          <div className="modal-info-container">
            <h2>{book.title}</h2>
            
            {book.author && <p><strong>Tác giả:</strong> {book.author}</p>}
            
            {book.description && <p><strong>Mô tả:</strong> {book.description}</p>}
            
            {book.product_url && (
              <a 
                href={book.product_url} 
                target="_blank" 
                rel="noopener noreferrer" 
                className="product-link"
              >
                Xem trên trang bán hàng
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default BookModal;