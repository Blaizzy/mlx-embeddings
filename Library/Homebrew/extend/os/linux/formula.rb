# typed: true
# frozen_string_literal: true

class Formula
  undef shared_library
  undef rpath
  undef deuniversalize_machos

  sig { params(name: String, version: T.nilable(T.any(String, Integer))).returns(String) }
  def shared_library(name, version = nil)
    suffix = if version == "*" || (name == "*" && version.blank?)
      "{,.*}"
    elsif version.present?
      ".#{version}"
    end
    "#{name}.so#{suffix}"
  end

  sig { returns(String) }
  def rpath
    "'$ORIGIN/../lib'"
  end

  sig { params(targets: T.nilable(T.any(Pathname, String))).void }
  def deuniversalize_machos(*targets); end
end
