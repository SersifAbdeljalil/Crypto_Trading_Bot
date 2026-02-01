import React, { useState, useEffect } from "react"
import axios from "axios"
import * as Icon from "react-cryptocoins"
import moment from "moment"
import styled from "styled-components"

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

// Crypto Icons definition
const cryptoIcons = {
  ETH: <Icon.Eth color={COLORS.ethereumPurple} />,
  LTC: <Icon.Ltc color={COLORS.blockchainBlue} />,
  XRP: <Icon.Xrp color={COLORS.blockchainBlue} />,
  BTC: <Icon.Btc color={COLORS.goldAccent} />,
}

const url = "http://localhost:5000/"

const Card = styled.div`
  padding: 24px;
  background-color: ${COLORS.cardBg};
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  transition: transform 0.2s ease-in-out, flex 0.2s ease-in-out,
    height 0.3s ease-in-out;
  margin-bottom: 20px;
  flex: 1;
  overflow: hidden;
  border: 1px solid ${COLORS.border};
`

const Title = styled.h2`
  margin-bottom: 20px;
  color: ${COLORS.textPrimary};
  font-size: 1.8rem;
  background: linear-gradient(135deg, ${COLORS.ethereumPurple}, ${COLORS.blockchainBlue});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`

const Header = styled.div`
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
`

const ToggleContainer = styled.div`
  display: flex;
  align-items: center;
  position: relative;
  width: 120px;
  height: 40px;
  background-color: ${COLORS.surface};
  border: 1px solid ${COLORS.border};
  border-radius: 20px;
  padding: 5px;
  cursor: pointer;
`

const ToggleSwitch = styled.div`
  position: absolute;
  top: 5px;
  left: ${props => (props.active ? "65px" : "5px")};
  width: 50px;
  height: 30px;
  background: linear-gradient(135deg, ${COLORS.ethereumPurple}, ${COLORS.blockchainBlue});
  border-radius: 15px;
  box-shadow: 0px 4px 12px rgba(98, 126, 234, 0.4);
  transition: left 0.3s;
`

const ToggleLabel = styled.span`
  font-size: 14px;
  color: ${props => (props.active ? COLORS.textPrimary : COLORS.textSecondary)};
  z-index: 1;
  flex: 1;
  text-align: center;
  font-weight: ${props => (props.active ? "bold" : "normal")};
`

const NewsItemContainer = styled.a`
  display: flex;
  align-items: center;
  padding: 14px;
  background-color: ${COLORS.surface};
  border: 1px solid ${COLORS.border};
  border-radius: 12px;
  margin-bottom: 12px;
  justify-content: space-between;
  text-decoration: none;
  color: inherit;
  transition: all 0.2s;

  &:hover {
    background-color: ${COLORS.border};
    transform: translateX(4px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
  }
`

const NewsDate = styled.div`
  color: ${COLORS.textSecondary};
  font-size: 12px;
  flex: 1.5;
`

const NewsTitle = styled.div`
  font-size: 14px;
  flex: 4;
  margin-left: 20px;
  margin-right: 20px;
  color: ${COLORS.textPrimary};
`

const NewsSentiment = styled.div`
  flex: 1;
  display: flex;
  justify-content: center;
  padding: 6px 12px;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: bold;
  color: ${props =>
    props.positive ? COLORS.cryptoGreen : props.negative ? COLORS.cryptoRed : COLORS.textSecondary};
  background-color: ${props =>
    props.positive
      ? "rgba(16, 185, 129, 0.15)"
      : props.negative
      ? "rgba(239, 68, 68, 0.15)"
      : "rgba(138, 146, 178, 0.15)"};
  border: 1px solid ${props =>
    props.positive ? COLORS.cryptoGreen : props.negative ? COLORS.cryptoRed : COLORS.border};
`

const Container = styled.div`
  display: flex;
  flex-direction: column;
  overflow: auto;
  padding: 0;
  background-color: transparent;
  border-radius: 10px;
  color: ${COLORS.textPrimary};
  font-family: Arial, sans-serif;

  &::-webkit-scrollbar {
    width: 8px;
  }

  &::-webkit-scrollbar-thumb {
    background: ${COLORS.border};
    border-radius: 10px;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: ${COLORS.ethereumLight};
  }
`

const TransactionContainer = styled.div`
  display: flex;
  align-items: center;
  padding: 14px;
  background-color: ${COLORS.surface};
  border: 1px solid ${COLORS.border};
  border-radius: 12px;
  margin-bottom: 12px;
  justify-content: space-between;
  transition: all 0.2s;

  &:hover {
    background-color: ${COLORS.border};
    transform: translateX(4px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
  }
`

const IconContainer = styled.div`
  display: flex;
  align-items: center;
  flex: 1;
`

const Text = styled.div`
  color: ${COLORS.textPrimary};
  margin-left: 10px;
  font-weight: bold;
`

const InfoLabel = styled.div`
  color: ${COLORS.textSecondary};
  font-size: 0.75rem;
  margin-top: 4px;
`

const SideContainer = styled.div`
  display: flex;
  align-items: center;
`

const SideIndicator = styled.div`
  width: 10px;
  height: 10px;
  background-color: ${props => (props.sell ? COLORS.cryptoRed : COLORS.cryptoGreen)};
  border-radius: 50%;
  margin-right: 8px;
  box-shadow: ${props => 
    props.sell 
      ? `0 0 10px ${COLORS.cryptoRed}` 
      : `0 0 10px ${COLORS.cryptoGreen}`};
`

const Info = styled.div`
  color: ${COLORS.textPrimary};
  font-size: 14px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-weight: 600;
`

const TransactionSectionContainer = styled(Card)`
  max-height: 590px;
  overflow-y: auto;

  &::-webkit-scrollbar {
    display: none;
  }

  -ms-overflow-style: none;
  scrollbar-width: none;
`

const NewsSectionContainer = styled(Card)`
  max-height: 590px;
  overflow-y: auto;

  &::-webkit-scrollbar {
    display: none;
  }

  -ms-overflow-style: none;
  scrollbar-width: none;
`

const MainContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  min-height: 100vh;
`

const CardContainer = styled.div`
  display: flex;
  flex-direction: column;
  width: 100%;
`

const TransactionSection = () => {
  const [transactionHistory, setTransactionHistory] = useState([])

  async function fetchData() {
    try {
      const response = await axios.get(`${url}/all_transaction_history/10`)
      setTransactionHistory(response.data)
    } catch (error) {
      console.error(error)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  return (
    <Container>
      <Title>Transaction History</Title>
      {transactionHistory.map(transaction => (
        <SingleTransactionContainer
          key={transaction.id}
          transaction={transaction}
        />
      ))}
    </Container>
  )
}

const SingleTransactionContainer = ({ transaction }) => {
  const icon_name = transaction.symbol

  return (
    <TransactionContainer>
      <IconContainer>
        {cryptoIcons[icon_name]}
        <Text>{transaction.symbol}</Text>
      </IconContainer>
      <Info>
        {transaction.price_with_fee.toFixed(2)}
        <InfoLabel>Price</InfoLabel>
      </Info>
      <Info>
        {parseFloat(transaction.qty).toFixed(2)}
        <InfoLabel>Qty</InfoLabel>
      </Info>
      <Info>
        {moment
          .unix(transaction.timestamp / 1000)
          .format("DD MMM YYYY hh:mm a")}
        <InfoLabel>Date & Time</InfoLabel>
      </Info>
      {transaction.side === "SELL" && (
        <Info
          style={{
            color: transaction.profits <= 0 ? COLORS.cryptoRed : COLORS.cryptoGreen,
          }}
        >
          {transaction.profits === "---"
            ? `${transaction.profits} (${transaction.profits_percent}%)`
            : `${parseFloat(transaction.profits).toFixed(2)}$ (${parseFloat(
                transaction.profits_percent
              ).toFixed(2)}%)`}
          <InfoLabel>Gains/Losses</InfoLabel>
        </Info>
      )}
      <SideContainer>
        <SideIndicator sell={transaction.side === "SELL"} />
        <Info>{transaction.side}</Info>
      </SideContainer>
    </TransactionContainer>
  )
}

const LeftSideSection = () => {
  const [newsFilter, setNewsFilter] = useState("top")
  const [topNews, setTopNews] = useState([])
  const [allNews, setAllNews] = useState([])
  const [newsData, setNewsData] = useState([])

  async function fetchTopNews() {
    try {
      const response = await axios.get(`${url}/news/top/8`)
      let data = response.data
      setTopNews(data)
      setNewsData(data)
    } catch (error) {
      console.error(error)
    }
  }

  async function fetchAllNews() {
    try {
      const response = await axios.get(`${url}/news/all/8`)
      let data = response.data
      setAllNews(data)
    } catch (error) {
      console.error(error)
    }
  }

  useEffect(() => {
    fetchTopNews()
    fetchAllNews()
  }, [])

  useEffect(() => {
    if (newsFilter === "top") {
      setNewsData(topNews)
    } else {
      setNewsData(allNews)
    }
  }, [newsFilter, topNews, allNews])

  return (
    <MainContainer>
      <CardContainer>
        <TransactionSectionContainer>
          <TransactionSection />
        </TransactionSectionContainer>
        <NewsSectionContainer>
          <Header>
            <Title>News</Title>
            <ToggleContainer
              onClick={() =>
                setNewsFilter(newsFilter === "top" ? "latest" : "top")
              }
            >
              <ToggleLabel active={newsFilter === "top"}>Top</ToggleLabel>
              <ToggleLabel active={newsFilter === "latest"}>Latest</ToggleLabel>
              <ToggleSwitch active={newsFilter === "latest"} />
            </ToggleContainer>
          </Header>
          {newsData.slice(0, 8).map((news, index) => (
            <NewsItemContainer key={index} href={news.link} target="_blank">
              <NewsDate>
                {moment(news.date).isValid()
                  ? moment(news.date).format("DD MMM YYYY HH:mm, UTC")
                  : news.date}
              </NewsDate>
              <NewsTitle>{news.title}</NewsTitle>
              <NewsSentiment
                positive={news.sentiment === "Positive"}
                negative={news.sentiment === "Negative"}
              >
                {news.sentiment}
              </NewsSentiment>
            </NewsItemContainer>
          ))}
        </NewsSectionContainer>
      </CardContainer>
    </MainContainer>
  )
}

export default LeftSideSection