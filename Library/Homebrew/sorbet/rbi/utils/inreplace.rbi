# typed: strict

module Utils::Inreplace
  include Kernel

  sig { params(paths: T::Array[T.untyped], before: T.nilable(String), after: T.nilable(String), audit_result: T::Boolean).void }
  def inreplace(paths, before = nil, after = nil, audit_result = true); end

end

class StringInreplaceExtension
  sig { params(before: String, after: String).returns(T.nilable(String)) }
  def sub!(before, after)
  end

  sig { params(before: T.nilable(String), after: T.nilable(String), audit_result: T::Boolean).returns(T.nilable(String)) }
  def gsub!(before, after, audit_result = true); end

  sig {params(flag: String, new_value: String).void}
  def change_make_var!(flag, new_value)
  end

  sig {params(flags: T::Array[String]).void}
  def remove_make_var!(flags)
  end

  sig {params(flag: String).returns(String)}
  def get_make_var(flag)
  end
end
