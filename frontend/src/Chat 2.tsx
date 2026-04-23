import { FormEvent, useState } from 'react'
import magIcon from './assets/mag.png'

type ChatProps = {
  onSearchTerm: (value: string) => void
}

function Chat({ onSearchTerm }: ChatProps): JSX.Element {
  const [value, setValue] = useState('')

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault()

    const trimmedValue = value.trim()
    if (!trimmedValue) return

    onSearchTerm(trimmedValue)
    setValue('')
  }

  return (
    <div className="chat-bar">
      <form className="input-row" onSubmit={handleSubmit}>
        <img src={magIcon} alt="" aria-hidden="true" />
        <input
          type="text"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="Paste an essay idea or thesis for AI search..."
          aria-label="AI essay helper prompt"
        />
        <button type="submit" disabled={value.trim() === ''}>
          Search with AI
        </button>
      </form>
    </div>
  )
}

export default Chat
