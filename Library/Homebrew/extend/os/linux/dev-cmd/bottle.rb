# typed: true
# frozen_string_literal: true

module Homebrew
  sig { params(args: T.untyped, mtime: String).returns([String, T::Array[String]]) }
  def self.setup_tar_and_args!(args, mtime)
    # Without --only-json-tab bottles are never reproducible
    return ["tar", tar_args].freeze unless args.only_json_tab?

    ["tar", reproducible_gnutar_args(mtime)].freeze
  end
end
