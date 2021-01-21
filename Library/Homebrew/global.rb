# typed: false
# frozen_string_literal: true

require_relative "load_path"

require "English"
require "json"
require "json/add/exception"
require "pathname"
require "ostruct"
require "pp"
require "forwardable"

require "rbconfig"

RUBY_PATH = Pathname.new(RbConfig.ruby).freeze
RUBY_BIN = RUBY_PATH.dirname.freeze

require "rubygems"
# Only require "core_ext" here to ensure we're only requiring the minimum of
# what we need.
require "active_support/core_ext/object/blank"
require "active_support/core_ext/numeric/time"
require "active_support/core_ext/object/try"
require "active_support/core_ext/array/access"
require "active_support/core_ext/string/inflections"
require "active_support/core_ext/array/conversions"
require "active_support/core_ext/hash/deep_merge"
require "active_support/core_ext/file/atomic"
require "active_support/core_ext/enumerable"
require "active_support/core_ext/string/exclude"
require "active_support/core_ext/string/indent"

I18n.backend.available_locales # Initialize locales so they can be overwritten.
I18n.backend.store_translations :en, support: { array: { last_word_connector: " and " } }

ActiveSupport::Inflector.inflections(:en) do |inflect|
  inflect.irregular "formula", "formulae"
  inflect.irregular "is", "are"
  inflect.irregular "it", "they"
end

require "utils/sorbet"

HOMEBREW_BOTTLE_DEFAULT_DOMAIN = ENV["HOMEBREW_BOTTLE_DEFAULT_DOMAIN"]
HOMEBREW_BREW_DEFAULT_GIT_REMOTE = ENV["HOMEBREW_BREW_DEFAULT_GIT_REMOTE"]
HOMEBREW_CORE_DEFAULT_GIT_REMOTE = ENV["HOMEBREW_CORE_DEFAULT_GIT_REMOTE"]
HOMEBREW_DEFAULT_CACHE = ENV["HOMEBREW_DEFAULT_CACHE"]
HOMEBREW_DEFAULT_LOGS = ENV["HOMEBREW_DEFAULT_LOGS"]
HOMEBREW_DEFAULT_TEMP = ENV["HOMEBREW_DEFAULT_TEMP"]
HOMEBREW_REQUIRED_RUBY_VERSION = ENV["HOMEBREW_REQUIRED_RUBY_VERSION"]

HOMEBREW_PRODUCT = ENV["HOMEBREW_PRODUCT"]
HOMEBREW_VERSION = ENV["HOMEBREW_VERSION"]
HOMEBREW_WWW = "https://brew.sh"

HOMEBREW_USER_AGENT_CURL = ENV["HOMEBREW_USER_AGENT_CURL"]
HOMEBREW_USER_AGENT_RUBY =
  "#{ENV["HOMEBREW_USER_AGENT"]} ruby/#{RUBY_VERSION}-p#{RUBY_PATCHLEVEL}"
HOMEBREW_USER_AGENT_FAKE_SAFARI =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/602.4.8 " \
  "(KHTML, like Gecko) Version/10.0.3 Safari/602.4.8"

HOMEBREW_DEFAULT_PREFIX = "/usr/local"
HOMEBREW_DEFAULT_REPOSITORY = "#{HOMEBREW_DEFAULT_PREFIX}/Homebrew"
HOMEBREW_MACOS_ARM_DEFAULT_PREFIX = "/opt/homebrew"
HOMEBREW_MACOS_ARM_DEFAULT_REPOSITORY = HOMEBREW_MACOS_ARM_DEFAULT_PREFIX
HOMEBREW_LINUX_DEFAULT_PREFIX = "/home/linuxbrew/.linuxbrew"
HOMEBREW_LINUX_DEFAULT_REPOSITORY = "#{HOMEBREW_LINUX_DEFAULT_PREFIX}/Homebrew"

HOMEBREW_PULL_API_REGEX =
  %r{https://api\.github\.com/repos/([\w-]+)/([\w-]+)?/pulls/(\d+)}.freeze
HOMEBREW_PULL_OR_COMMIT_URL_REGEX =
  %r[https://github\.com/([\w-]+)/([\w-]+)?/(?:pull/(\d+)|commit/[0-9a-fA-F]{4,40})].freeze
HOMEBREW_RELEASES_URL_REGEX =
  %r{https://github\.com/([\w-]+)/([\w-]+)?/releases/download/(.+)}.freeze

require "fileutils"

require "os"
require "os/global"
require "messages"

module Homebrew
  extend FileUtils

  DEFAULT_PREFIX ||= HOMEBREW_DEFAULT_PREFIX
  DEFAULT_REPOSITORY ||= HOMEBREW_DEFAULT_REPOSITORY
  DEFAULT_CELLAR = "#{DEFAULT_PREFIX}/Cellar"
  DEFAULT_MACOS_CELLAR = "#{HOMEBREW_DEFAULT_PREFIX}/Cellar"
  DEFAULT_MACOS_ARM_CELLAR = "#{HOMEBREW_MACOS_ARM_DEFAULT_PREFIX}/Cellar"
  DEFAULT_LINUX_CELLAR = "#{HOMEBREW_LINUX_DEFAULT_PREFIX}/Cellar"

  class << self
    attr_writer :failed, :raise_deprecation_exceptions, :auditing

    def default_prefix?(prefix = HOMEBREW_PREFIX)
      prefix.to_s == DEFAULT_PREFIX
    end

    def failed?
      @failed ||= false
      @failed == true
    end

    def messages
      @messages ||= Messages.new
    end

    def raise_deprecation_exceptions?
      @raise_deprecation_exceptions == true
    end

    def auditing?
      @auditing == true
    end
  end
end

require "env_config"

require "config"
require "context"
require "extend/pathname"
require "extend/predicable"
require "extend/module"
require "cli/args"

require "PATH"

ENV["HOMEBREW_PATH"] ||= ENV["PATH"]
ORIGINAL_PATHS = PATH.new(ENV["HOMEBREW_PATH"]).map do |p|
  Pathname.new(p).expand_path
rescue
  nil
end.compact.freeze

require "set"

require "extend/string"

require "system_command"
require "exceptions"
require "utils"

require "official_taps"
require "tap"
require "tap_constants"

require "compat" unless Homebrew::EnvConfig.no_compat?
