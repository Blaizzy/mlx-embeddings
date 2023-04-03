# typed: true
# frozen_string_literal: true

module TimeRemaining
  refine Time do
    def remaining
      T.bind(self, Time)
      [0, self - Time.now].max
    end

    def remaining!
      r = remaining

      Kernel.raise Timeout::Error if r <= 0

      r
    end
  end
end

class Time
  # Backwards compatibility for formulae that used this ActiveSupport extension
  alias rfc3339 xmlschema
end
