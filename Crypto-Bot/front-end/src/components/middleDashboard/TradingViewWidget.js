import React, { useEffect, useRef, memo } from 'react';

const TradingViewWidget = ({ symbol }) => {
  const container = useRef();
  
  useEffect(() => {
    // Clear the container before adding a new script
    if (container.current) {
      container.current.innerHTML = "";
    }
    
    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = JSON.stringify({
      "autosize": false,
      "symbol": symbol,
      "width": "100%",
      "height": "400",
      "interval": "D",
      "timezone": "Etc/UTC",
      "theme": "dark", // Changed to dark theme to match Ethereum palette
      "style": "1",
      "locale": "en",
      "allow_symbol_change": true,
      "calendar": false,
      "support_host": "https://www.tradingview.com"
    });
    
    container.current.appendChild(script);
  }, [symbol]);
  
  return (
    <div 
      className="tradingview-widget-container" 
      ref={container} 
      style={{ 
        height: "400px", 
        width: "100%",
        backgroundColor: "#1A1B3A",
        borderRadius: "16px",
        overflow: "hidden",
        border: "1px solid #2D2E4E",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.4)",
        marginBottom: "20px"
      }}
    >
      <div 
        className="tradingview-widget-container__widget" 
        style={{ 
          height: "400px", 
          width: "100%" 
        }}
      ></div>
      <div 
        className="tradingview-widget-copyright"
        style={{
          position: "absolute",
          bottom: "10px",
          left: "10px",
          fontSize: "11px"
        }}
      >
        <a 
          href="https://www.tradingview.com/" 
          rel="noopener nofollow" 
          target="_blank"
          style={{
            color: "#627EEA",
            textDecoration: "none"
          }}
        >
          <span className="blue-text">Track all markets on TradingView</span>
        </a>
      </div>
    </div>
  );
};

export default memo(TradingViewWidget);