import { useEffect, useRef } from 'react'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartOptions,
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

interface StockChartProps {
  symbol: string
  height?: number
  showVolume?: boolean
}

const StockChart = ({ symbol, height = 400, showVolume = false }: StockChartProps) => {
  const chartRef = useRef<ChartJS<'line'>>(null)

  // Mock data - replace with real data from API
  const generateMockData = () => {
    const labels = []
    const prices = []
    const volumes = []
    const basePrice = 100
    
    for (let i = 0; i < 50; i++) {
      const date = new Date()
      date.setHours(date.getHours() - (50 - i))
      labels.push(date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
      
      const randomChange = (Math.random() - 0.5) * 2
      const price = basePrice + randomChange + Math.sin(i / 10) * 5
      prices.push(price)
      volumes.push(Math.random() * 1000000)
    }
    
    return { labels, prices, volumes }
  }

  const { labels, prices, volumes } = generateMockData()

  const data = {
    labels,
    datasets: [
      {
        label: symbol,
        data: prices,
        borderColor: 'rgb(14, 165, 233)',
        backgroundColor: 'rgba(14, 165, 233, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.4,
        fill: true,
      },
    ],
  }

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: 'rgba(26, 31, 46, 0.95)',
        titleColor: '#e5e7eb',
        bodyColor: '#9ca3af',
        borderColor: '#2a3441',
        borderWidth: 1,
        padding: 12,
        displayColors: false,
        callbacks: {
          label: (context) => {
            const label = context.dataset.label || ''
            const value = context.parsed.y
            return `${label}: $${value.toFixed(2)}`
          },
        },
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(42, 52, 65, 0.5)',
          drawBorder: false,
        },
        ticks: {
          color: '#6b7280',
          maxRotation: 0,
          autoSkipPadding: 20,
        },
      },
      y: {
        position: 'right',
        grid: {
          color: 'rgba(42, 52, 65, 0.5)',
          drawBorder: false,
        },
        ticks: {
          color: '#6b7280',
          callback: (value) => `$${value}`,
        },
      },
    },
  }

  useEffect(() => {
    // Add gradient to chart
    if (chartRef.current) {
      const chart = chartRef.current
      const ctx = chart.ctx
      const gradient = ctx.createLinearGradient(0, 0, 0, height)
      gradient.addColorStop(0, 'rgba(14, 165, 233, 0.3)')
      gradient.addColorStop(1, 'rgba(14, 165, 233, 0)')
      
      chart.data.datasets[0].backgroundColor = gradient
      chart.update()
    }
  }, [height])

  return (
    <div className="w-full" style={{ height }}>
      <Line ref={chartRef} data={data} options={options} />
    </div>
  )
}

export default StockChart