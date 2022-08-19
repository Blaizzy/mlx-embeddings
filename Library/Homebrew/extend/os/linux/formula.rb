# typed: true
# frozen_string_literal: true

class Formula
  undef shared_library
  undef loader_path
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
  def loader_path
    "$ORIGIN"
  end

  sig { params(targets: T.nilable(T.any(Pathname, String))).void }
  def deuniversalize_machos(*targets); end
end
