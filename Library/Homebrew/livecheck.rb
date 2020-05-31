# frozen_string_literal: true

# Livecheck can be used to check for newer versions of the software.
# The livecheck DSL specified in the formula is evaluated the methods
# of this class, which set the instance variables accordingly. The
# information is used by brew livecheck when checking for newer versions
# of the software.
class Livecheck
  # The reason for skipping livecheck for the formula.
  # e.g. `Not maintained`
  attr_reader :skip_msg

  def initialize(formula)
    @formula = formula
    @regex = nil
    @skip = false
    @skip_msg = nil
    @url = nil
  end

  # Sets the regex instance variable to the argument given, returns the
  # regex instance variable when no argument is given.
  def regex(pattern = nil)
    return @regex if pattern.nil?

    @regex = pattern
  end

  # Sets the skip instance variable to true, indicating that livecheck
  # must be skipped for the formula. If an argument is given and present,
  # its value is assigned to the skip_msg instance variable, else nil is
  # assigned.
  def skip(skip_msg = nil)
    @skip = true
    @skip_msg = skip_msg.presence
  end

  # Should livecheck be skipped for the formula?
  def skip?
    @skip
  end

  # Sets the url instance variable to the argument given, returns the url
  # instance variable when no argument is given.
  def url(val = nil)
    return @url if val.nil?

    @url = case val
    when :head, :stable, :devel
      @formula.send(val).url
    when :homepage
      @formula.homepage
    else
      val
    end
  end

  # Returns a Hash of all instance variable values.
  def to_hash
    {
      "regex"    => @regex,
      "skip"     => @skip,
      "skip_msg" => @skip_msg,
      "url"      => @url,
    }
  end
end
