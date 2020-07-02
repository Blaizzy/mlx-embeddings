# typed: strict

module Tty
  include Kernel

  sig{ params(string: String).returns(String) }
  def strip_ansi(string)
  end

  sig{ returns(Integer) }
  def width()
  end

  sig{ params(string: String).returns(T.nilable(String)) }
  def truncate(string)
  end

  def append_to_escape_sequence(code)
  end

  sig{ returns(String) }
  def current_escape_sequence()
  end

  sig{ void }
  def reset_escape_sequence!()
  end

  sig{ returns(String) }
  def to_s
  end

  sig { returns(T::Boolean) }
  def color?
  end
end
