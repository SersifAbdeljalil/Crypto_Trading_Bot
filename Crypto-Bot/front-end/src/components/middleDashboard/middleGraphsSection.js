import React, { useState, useEffect } from "react";
import Binance from "binance-api-node";
import styled from "styled-components";
import MenuCard from "./MenuCard";
import TradingViewWidget from "./TradingViewWidget";

const client = Binance();

// Ethereum-inspired color palette
const COLORS = {
  ethereumPurple: '#627EEA',
  ethereumDark: '#1C1C3C',
  ethereumLight: '#8A92B2',
  blockchainBlue: '#00D4FF',
  cryptoGreen: '#10B981',
  cryptoRed: '#EF4444',
  goldAccent: '#F59E0B',
  darkBg: '#0F0F23',
  cardBg: '#1A1B3A',
  surface: '#252641',
  border: '#2D2E4E',
  textPrimary: '#E5E7EB',
  textSecondary: '#9CA3AF',
}

const MiddleGraphsSection = () => {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [dailyStatsForSymbol, setDailyStatsForSymbol] = useState({});

  useEffect(() => {
    client.dailyStats({ symbol }).then(stat => {
      setDailyStatsForSymbol(stat);
    }).catch(error => {
      console.error('Error fetching daily stats:', error);
    });
  }, [symbol]);

  const dailyHigh = parseFloat(dailyStatsForSymbol.highPrice || 0).toFixed(2);
  const dailyLow = parseFloat(dailyStatsForSymbol.lowPrice || 0).toFixed(2);
  const priceChangePercent = dailyStatsForSymbol.priceChangePercent || 0;
  const priceChange = parseFloat(dailyStatsForSymbol.priceChange || 0).toFixed(2);
  const lastPrice = parseFloat(dailyStatsForSymbol.lastPrice || 0).toFixed(2);

  return (
    <Section>
      <TopHeaderSection
        setSymbol={setSymbol}
        symbol={symbol}
        dailyHigh={dailyHigh}
        dailyLow={dailyLow}
        priceChangePercent={priceChangePercent}
        lastPrice={lastPrice}
        priceChange={priceChange}
      />
      <ChartSection symbol={symbol} />
      <MenuCardSection>
        <MenuCard />
      </MenuCardSection>
    </Section>
  );
};

const TopHeaderSection = ({
  setSymbol,
  symbol,
  dailyHigh,
  dailyLow,
  priceChangePercent,
  lastPrice,
  priceChange,
}) => {
  const handleTabChange = tab => {
    setSymbol(tab);
  };

  return (
    <Header>
      <ToggleSwitch>
        <ToggleOption
          active={symbol === "BTCUSDT"}
          onClick={() => handleTabChange("BTCUSDT")}
        >
          🪙 BTC/USD
        </ToggleOption>
        <ToggleOption
          active={symbol === "ETHUSDT"}
          onClick={() => handleTabChange("ETHUSDT")}
        >
          💎 ETH/USD
        </ToggleOption>
        <ToggleOption
          active={symbol === "XRPUSDT"}
          onClick={() => handleTabChange("XRPUSDT")}
        >
          ⚡ XRP/USD
        </ToggleOption>
      </ToggleSwitch>
      <DataContainer>
        <DataCard>
          <GreyText>Last Price</GreyText>
          <DataText
            style={{ color: priceChangePercent >= 0 ? COLORS.cryptoGreen : COLORS.cryptoRed }}
          >
            ${lastPrice}
          </DataText>
        </DataCard>
        <DataCard>
          <GreyText>24h Change</GreyText>
          <DataChange>
            <DataText
              style={{ color: priceChangePercent >= 0 ? COLORS.cryptoGreen : COLORS.cryptoRed }}
            >
              ${priceChange}
            </DataText>
            <div style={{ marginLeft: '8px' }}>
              {priceChangePercent >= 0 && (
                <ChangeSymbol style={{ color: COLORS.cryptoGreen }}>+</ChangeSymbol>
              )}
              <DataText
                style={{
                  color: priceChangePercent >= 0 ? COLORS.cryptoGreen : COLORS.cryptoRed,
                }}
              >
                {priceChangePercent}%
              </DataText>
            </div>
          </DataChange>
        </DataCard>
        <DataCard>
          <GreyText>24h High</GreyText>
          <DataText>${dailyHigh}</DataText>
        </DataCard>
        <DataCard>
          <GreyText>24h Low</GreyText>
          <DataText>${dailyLow}</DataText>
        </DataCard>
      </DataContainer>
    </Header>
  );
};

const ChartSection = ({ symbol }) => {
  return symbol ? <TradingViewWidget symbol={symbol} /> : null;
};

export default MiddleGraphsSection;

const Section = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  background-color: ${COLORS.darkBg};
  border-radius: 0px;
  font-family: "Roboto", sans-serif;
`;

const Header = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  padding: 24px;
  background-color: ${COLORS.cardBg};
  border: 1px solid ${COLORS.border};
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  margin-bottom: 20px;
`;

const ToggleSwitch = styled.div`
  display: flex;
  background-color: ${COLORS.surface};
  border: 1px solid ${COLORS.border};
  border-radius: 24px;
  padding: 6px;
  margin-bottom: 24px;
  gap: 4px;
`;

const ToggleOption = styled.div`
  padding: 12px 24px;
  border-radius: 20px;
  cursor: pointer;
  background: ${props => 
    props.active 
      ? `linear-gradient(135deg, ${COLORS.ethereumPurple}, ${COLORS.blockchainBlue})` 
      : "transparent"};
  box-shadow: ${props => 
    props.active 
      ? `0 4px 16px rgba(98, 126, 234, 0.4)` 
      : "none"};
  color: ${props => (props.active ? "#ffffff" : COLORS.textSecondary)};
  font-weight: ${props => (props.active ? "bold" : "normal")};
  transition: all 0.3s ease;

  &:hover {
    background: ${props => 
      props.active 
        ? `linear-gradient(135deg, ${COLORS.ethereumPurple}, ${COLORS.blockchainBlue})` 
        : COLORS.border};
    color: ${props => (props.active ? "#ffffff" : COLORS.textPrimary)};
  }
`;

const DataContainer = styled.div`
  display: flex;
  justify-content: space-around;
  width: 100%;
  gap: 12px;
`;

const DataCard = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px 30px;
  background-color: ${COLORS.surface};
  border: 1px solid ${COLORS.border};
  border-radius: 12px;
  margin: 10px;
  flex: 1;
  transition: all 0.2s ease;

  &:hover {
    background-color: ${COLORS.border};
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
  }
`;

const GreyText = styled.span`
  color: ${COLORS.textSecondary};
  font-size: 14px;
  margin-bottom: 8px;
`;

const DataText = styled.span`
  color: ${COLORS.textPrimary};
  font-size: 20px;
  font-weight: bold;
`;

const DataChange = styled.div`
  display: flex;
  align-items: center;
`;

const ChangeSymbol = styled.span`
  font-size: 18px;
  margin-right: 4px;
  font-weight: bold;
`;

const ChartContainer = styled.div`
  width: 100%;
  height: 400px;
  background-color: ${COLORS.cardBg};
  border: 1px solid ${COLORS.border};
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  margin-bottom: 20px;
`;

const MenuCardSection = styled.div`
  width: 100%;
  padding: 24px;
  background-color: ${COLORS.cardBg};
  border: 1px solid ${COLORS.border};
  border-radius: 16px;
  height: 550px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  font-family: "Roboto", sans-serif;
`;