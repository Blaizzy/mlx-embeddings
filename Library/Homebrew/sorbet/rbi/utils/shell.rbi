# typed: strict

module Utils::Shell
  include Kernel

  sig{ params(path: String).returns(T.nilable(Symbol)) }
  def from_path(path)
  end

  sig{ returns(T.nilable(Symbol)) }
  def preferred
  end

  def parent
  end

  def export_value(key, value, shell = preferred)
  end

  sig{ returns(String) }
  def profile
  end

  def set_variable_in_profile(variable, value)
  end

  sig{ params(path: String).returns(T.nilable(String)) }
  def prepend_path_in_profile(path)
  end

  sig{ params(str: String).returns(T.nilable(String)) }
  def csh_quote(str)
  end

  sig{ params(str: String).returns(T.nilable(String)) }
  def sh_quote(str)
  end
end
