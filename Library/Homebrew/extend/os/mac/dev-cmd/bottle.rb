# typed: true
# frozen_string_literal: true

module Homebrew
  sig { returns(T::Array[String]) }
  def self.tar_args
    ["--no-mac-metadata", "--no-acls", "--no-xattrs"].freeze
  end
end
