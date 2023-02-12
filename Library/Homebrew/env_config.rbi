# typed: strict

module Homebrew::EnvConfig
  # This is necessary due to https://github.com/sorbet/sorbet/issues/6726
  sig { returns(String) }
  def self.api_auto_update_secs; end
end
