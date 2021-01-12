# typed: true
# frozen_string_literal: true

# The {Livecheck} class implements the DSL methods used in a formula's or cask's
# `livecheck` block and stores related instance variables. Most of these methods
# also return the related instance variable when no argument is provided.
#
# This information is used by the `brew livecheck` command to control its
# behavior. Example `livecheck` blocks can be found in the
# [`brew livecheck` documentation](https://docs.brew.sh/Brew-Livecheck).
class Livecheck
  extend Forwardable

  # A very brief description of why the formula/cask is skipped (e.g. `No longer
  # developed or maintained`).
  # @return [String, nil]
  attr_reader :skip_msg

  def initialize(formula_or_cask)
    @formula_or_cask = formula_or_cask
    @regex = nil
    @skip = false
    @skip_msg = nil
    @strategy = nil
    @url = nil
  end

  # Sets the `@regex` instance variable to the provided `Regexp` or returns the
  # `@regex` instance variable when no argument is provided.
  #
  # @param pattern [Regexp] regex to use for matching versions in content
  # @return [Regexp, nil]
  def regex(pattern = nil)
    case pattern
    when nil
      @regex
    when Regexp
      @regex = pattern
    else
      raise TypeError, "Livecheck#regex expects a Regexp"
    end
  end

  # Sets the `@skip` instance variable to `true` and sets the `@skip_msg`
  # instance variable if a `String` is provided. `@skip` is used to indicate
  # that the formula/cask should be skipped and the `skip_msg` very briefly
  # describes why it is skipped (e.g. "No longer developed or maintained").
  #
  # @param skip_msg [String] string describing why the formula/cask is skipped
  # @return [Boolean]
  def skip(skip_msg = nil)
    if skip_msg.is_a?(String)
      @skip_msg = skip_msg
    elsif skip_msg.present?
      raise TypeError, "Livecheck#skip expects a String"
    end

    @skip = true
  end

  # Should `livecheck` skip this formula/cask?
  def skip?
    @skip
  end

  # Sets the `@strategy` instance variable to the provided `Symbol` or returns
  # the `@strategy` instance variable when no argument is provided. The strategy
  # symbols use snake case (e.g. `:page_match`) and correspond to the strategy
  # file name.
  #
  # @param symbol [Symbol] symbol for the desired strategy
  # @return [Symbol, nil]
  def strategy(symbol = nil, &block)
    @strategy_block = block if block

    case symbol
    when nil
      @strategy
    when Symbol
      @strategy = symbol
    else
      raise TypeError, "Livecheck#strategy expects a Symbol"
    end
  end

  attr_reader :strategy_block

  # Sets the `@url` instance variable to the provided argument or returns the
  # `@url` instance variable when no argument is provided. The argument can be
  # a `String` (a URL) or a supported `Symbol` corresponding to a URL in the
  # formula/cask (e.g. `:stable`, `:homepage`, `:head`, `:url`).
  # @param val [String, Symbol] URL to check for version information
  # @return [String, nil]
  def url(val = nil)
    case val
    when nil
      @url
    when String, :head, :homepage, :stable, :url
      @url = val
    else
      raise TypeError, "Livecheck#url expects a String or valid Symbol"
    end
  end

  delegate version: :@formula_or_cask
  private :version

  # Returns a `Hash` of all instance variable values.
  # @return [Hash]
  def to_hash
    {
      "regex"    => @regex,
      "skip"     => @skip,
      "skip_msg" => @skip_msg,
      "strategy" => @strategy,
      "url"      => @url,
    }
  end
end
